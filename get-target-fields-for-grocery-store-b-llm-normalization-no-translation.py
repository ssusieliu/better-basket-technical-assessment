import json
import os
import re
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError, ClientError
import asyncio
from asynciolimiter import LeakyBucketLimiter
import html
from bs4 import BeautifulSoup
from datetime import datetime

chunk_size = 400 # Assuming ~10 tokens per item, max output tokens of ~8K, and adding some buffer

async def inference_brand(parsed_results_chunk: list, store_a_brands) -> list:
    """
    Takes in a list of parsed results and inferences brand
    Parameters: parsed results chunk, LLM client, LLM cached content
    Returns: dict of SKU: inferenced brand for each product in chunk
    """
    max_retries = 5
    retry_delay = 2
    attempt = 0

    while attempt < max_retries:
        try:
        # Initialize client here so we can create shared cached context
            client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY"),
            )
            model_name = "gemini-2.0-flash-lite"
            
            prompt = f"""You are a merchandiser at a large South American grocery store chain who has expertise on the brands your store carries. 
            I have a list of brands carried at a competing grocery store:
            {str(store_a_brands)}

            You have the following dictionaries of grocery product data for your store, which includes product names but not product brands.
            {str(parsed_results_chunk)}

            Identify the brand name of the product based on its product name, your knowledge of existing grocery product brands, and the list of brands carried at the competing store.
            If the same brand is carried at the competing grocery store, MAKE SURE to match the EXACT SPELLING of the brand name that the competing store uses, including spaces and special characters.
            
            Some product names are in Spanish - please adjust your thinking accordingly. 
            
            If the product appears to have no brand, return NO BRAND instead of UNKNOWN.
            Output the results in a json formatted containing a single dictionary where the keys are the SKUs and the values are the corresponding brands
            """
            
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]

            print(f'{datetime.now()} - Submitting another API request')
            
            api_response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=1,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192,
                    response_mime_type="text/plain"
                    ),
            )

            api_response_text = api_response.candidates[0].content.parts[0].text

            json_start = api_response_text.find('{')
            json_end = api_response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_response = api_response_text[json_start:json_end]
                results = json.loads(json_response)
                return results
            else:
                print(f"Could not find JSON in LLM's response. Start of response {api_response_text[:10]} End of response {api_response_text[len(api_response_text)-10:]}")
                return []
        
        except (ClientError, ServerError, APIError) as e:
            attempt += 1
            if attempt >= max_retries:
                print(f"Failed after {max_retries} attempts: {e}")
                return []
            
            wait_time = retry_delay * (2 ** (attempt - 1))
            await asyncio.sleep(wait_time)
            print(f'Attempt {attempt} failed with error {e}. Retrying in {wait_time} seconds')

        except Exception as e:
            print(f'Unexpected error: {e}')
            return []

async def inference_brand_with_rate_limiting_and_caching(parsed_results: list, reference_data_filepath: str):
    """
    Initializes LLM client and inferences brand for products in parsed results in chunks
    Parameters: parsed results, reference file datapath
    Returns: dict of SKU: inferenced brand for each product
    """
    rate_limiter = LeakyBucketLimiter(rate = 14/60, capacity = 14) # Gemini has a rate limit of 15 requests per min. Using 14 to create some buffer
    api_responses = []
    all_responses = {}
    coroutines = []

    # Extract list of brands from reference data so we can pass it into the prompt
    with open(reference_data_filepath, 'r', encoding = 'utf-8') as f:
        reference_data = json.load(f)
    
    store_a_brands = set()
    for product in reference_data:
        if product['brand'] not in store_a_brands:
            store_a_brands.add(product['brand'])

    for i in range(0, len(parsed_results), chunk_size):
        # Chunk requests to avoid hitting rate limits
        parsed_results_chunk = [{item['sku']: item['product_name']} for item in parsed_results[i: i + chunk_size]] # Only need to pass in product name and SKU
        coroutines.append(inference_brand(parsed_results_chunk, store_a_brands))

    api_responses = await asyncio.gather(*map(rate_limiter.wrap, coroutines))
    
    for response in api_responses:
        if response: # Skip empty responses from failed requests
            all_responses.update(response)

    return all_responses

def extract_fields_from_parsed_html(product: str):
    """Parses product details using HTML tags"""
    # Extract product name
    name = None
    name_link = product.find('a', title=True)
    if name_link:
        name = name_link.get('title')
    
    if not name:
        name_h3 = product.find('h3')
        if name_h3 and name_h3.find('a'):
            name = name_h3.find('a').get_text().strip()
    
    # Extract SKU - use this as a product ID to link inference results to the parsed results. Use the parsed results where we can to prevent messing up the data via LLM hallucination
    sku = None
    sku_input = product.find('input', {'name': 'sku'})
    sku = sku_input.get('value') if sku_input else None
    
    # Extract brand but get LLM to guess as well since some brands will have multiple words in the title
    words = name.split()
    brand = words[0] if words else None

    # Extract size
    size_tag = product.find('p', class_='text-center text-muted')
    size = size_tag.get_text().strip() if size_tag else None
    
    # Extract price
    price_tag = product.find('p', class_='text-center precio')
    price = price_tag.get_text().strip() if price_tag else None

    # Normalize price
    # Handle multi-buy deals (e.g., "2/$6.00")
    multi_buy_deal = None
    if "/" in price:
        multi_buy_match = re.match(r"(\d+)/\$(\d+\.\d+|\d+)", price)
        if multi_buy_match:
            quantity = int(multi_buy_match.group(1))
            total_price = float(multi_buy_match.group(2))
            unit_price = total_price / quantity
            
            # Add multi-buy field and normalize price
            multi_buy_deal = price # Add this to contextualize price difference
            price = unit_price
    
    # Handle cent symbol (e.g., "95¢")
    elif "\u00a2" in price:
        cents = float(re.search(r"(\d+)\u00a2", price).group(1))
        price = float(cents/100)
        
    # Handle prices with "LB" or other units
    elif "LB" in price:
        # Extract just the dollar amount
        dollar_match = re.search(r"\$(\d+\.\d+|\d+)", price)
        if dollar_match:
            price = float(dollar_match.group(1))
            
    # Handle plain dollar amounts
    elif price.startswith("$"):
        # Ensure proper formatting
        dollar_match = re.search(r"\$(\d+\.\d+|\d+)", price)
        if dollar_match:
            price = float(dollar_match.group(1))
    
    # See if we can extract quantity 
    quantity = None
    quantity_match = re.search(r'\b(\d+)\s*(?:count|ct|pk|pc|piece|roll|pack)?s?\b', name, re.IGNORECASE)
    if quantity_match:
        count = quantity_match.group(1)
        quantity = count
    if name:
        return {
            "sku": sku,
            "product_name": name,
            "brand": brand,
            "size": size,
            "price": price,
            "quantity": quantity,
            "multi-buy deal": multi_buy_deal
        }

def process_html_item(item):
    """Process a single HTML item and extract product data"""
    results = []

    # Check if item has html_data
    if not (isinstance(item, dict) and 'data' in item and 'html_data' in item['data']):
        return results
    
    html_content = item['data']['html_data']

    # Decode HTML entities
    html_content = html.unescape(html_content)

    # Parse w BeautifulSoup
    soup = BeautifulSoup(html_content, features="html.parser")

    # Find product blocks
    product_blocks = soup.find_all('div', class_ = 'product-grid-item')
    print(f"Found {len(product_blocks)} product blocks in this item")
    
    for product in product_blocks:
        extracted_fields = extract_fields_from_parsed_html(product)
        results.append(extracted_fields)

    return results

async def extract_and_save_products(file_path, output_path, reference_data_filepath):
    try:
        extracted_product_fields = []
        # Initialize field counting variables
        total_products_found = 0
        field_counts = {
            "sku": 0,
            "product_name": 0,
            "brand": 0,
            "size": 0,
            "price": 0, 
            "quantity": 0,
            "multi-buy deal": 0
        }
        
        # Load the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"Successfully loaded JSON with {len(data)} items")
        
        html_parse_start_time = datetime.now()
        # Parse HTML and extract name, size, and price
        for item in data:
            extracted_fields = process_html_item(item)
            extracted_product_fields.extend(extracted_fields)
        html_parse_end_time = datetime.now()
        parsing_time = (html_parse_end_time-html_parse_start_time).seconds
        
        # Count fields that were successfully extracted during HTML parsing
        for product in extracted_product_fields:
            total_products_found += 1
            for field in field_counts.keys():
                if field in product and product[field] is not None:
                    field_counts[field] += 1
        
        print(f"Extracted information for {len(extracted_product_fields)} products. Parsing took {parsing_time} seconds")
        
        # Inference brand and quantity + clean up price data using a batched LLM call
        inference_start_time = datetime.now()
        inference_results = await inference_brand_with_rate_limiting_and_caching(extracted_product_fields, reference_data_filepath)
        inference_end_time = datetime.now()
        inference_time = (inference_end_time - inference_start_time).seconds

        print(f'Inferencing took {inference_time} seconds')

        # Track LLM inference success rate
        inference_success_count = 0
        for product in extracted_product_fields:
            sku = product['sku']
            inferenced_brand = inference_results.get(sku)
            if inferenced_brand:
                product['brand'] = inferenced_brand
                inference_success_count += 1
        
        # Count fields again after LLM inference to update brand field counts
        field_counts["brand"] = 0  # Reset brand count since we're replacing it
        for product in extracted_product_fields:
            if "brand" in product and product["brand"] is not None:
                field_counts["brand"] += 1
        
        # Write to JSON file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(extracted_product_fields, outfile, indent=2)
        
        # Print comprehensive statistics summary
        print("\n" + "="*60)
        print(" STORE B DATA EXTRACTION SUMMARY ")
        print("="*60)
        
        print(f"\nTotal HTML items processed: {len(data)}")
        print(f"Total product blocks found: {total_products_found}")
        print(f"Products with complete data: {len(extracted_product_fields)}")
        
        print("\nField population statistics:")
        for field, count in field_counts.items():
            percentage = (count / total_products_found) * 100 if total_products_found > 0 else 0
            print(f"  - {field:<12}: {count:>6} / {total_products_found} ({percentage:.1f}%)")
        
        print(f"\nLLM brand inference:")
        print(f"  - Total products sent for inference: {len(extracted_product_fields)}")
        print(f"  - Products with successful brand inference: {inference_success_count}")
        print(f"  - Success rate: {(inference_success_count / len(extracted_product_fields) * 100):.1f}%")
        
        print(f"\nPerformance metrics:")
        print(f"  - HTML parsing time: {parsing_time} seconds")
        print(f"  - LLM inference time: {inference_time} seconds")
        print(f"  - Total processing time: {parsing_time + inference_time} seconds")
        
        print(f"\nOutput saved to: {output_path}")
        print("="*60)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Update the path to your file
asyncio.run(extract_and_save_products(r"C:\Users\Susie's PC\better-basket-technical-assessment\grocery_store_b.json", 
                         r"C:\Users\Susie's PC\better-basket-technical-assessment\grocery_store_b_relevant_fields.json",
                         r"C:\Users\Susie's PC\better-basket-technical-assessment\grocery_store_a_relevant_fields.json"))