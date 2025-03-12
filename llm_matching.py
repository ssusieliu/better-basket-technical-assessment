import os
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError, APIError
import asyncio
from asynciolimiter import LeakyBucketLimiter
import json
import re
from datetime import datetime

# Helper functions
def extract_json_from_text(text):
    """Extract JSON array from LLM response text."""
    json_start = text.find('[')
    json_end = text.rfind(']') + 1
    
    if json_start >= 0 and json_end > json_start + 1:
        try:
            return json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            print("Failed to parse JSON from response")
    return []

def format_price_diff(value, is_percentage=False):
    """Format price difference with sign and proper decimals."""
    if is_percentage:
        return f"+{value:.1f}%" if value >= 0 else f"-{abs(value):.1f}%"
    return f"+${value:.2f}" if value >= 0 else f"-${abs(value):.2f}"

def save_results(matches, path, message="Saved results"):
    """Save results to a JSON file with error handling."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(matches, f, indent=2)
        print(f"{message} to {path}")
    except Exception as e:
        print(f"Error saving to {path}: {e}")

def extract_numeric_value(size_str):
    """Extract numeric value from a size string (e.g., "20 oz" -> 20)"""
    if not size_str:
        return None
    
    match = re.search(r'(\d+(?:\.\d+)?|\.\d+)', size_str)
    return float(match.group(1)) if match else None

# Group products by brand to get small, manageable lists of products to compare
def get_overlapping_brands_and_their_products(file_path_a, file_path_b):
    try:
        with open(file_path_a, 'r') as f_a, open(file_path_b, 'r') as f_b:
            store_a_data = json.load(f_a)
            store_b_data = json.load(f_b)
    except Exception as e:
        print(f"Error loading product data: {e}")
        return {}

    store_a_brands = {}
    matching_brands = set()
    matching_brand_products_count = 0

    # Process store A products
    for product in store_a_data:
        brand = product['brand']
        if brand not in store_a_brands:
            store_a_brands[brand] = {'store_a_products': [], 'store_b_products': []}
        store_a_brands[brand]['store_a_products'].append(product)

    # Process store B products and find matches
    for product in store_b_data:
        brand = product['brand']
        if brand in store_a_brands:
            if not store_a_brands[brand]['store_b_products']:
                matching_brands.add(brand)
            store_a_brands[brand]['store_b_products'].append(product)
            matching_brand_products_count += 1

    print(f'Found {len(matching_brands)} matching brands with {matching_brand_products_count} products')

    # Filter to only brands present in both stores
    return {brand: store_a_brands[brand] for brand in matching_brands}

async def find_matching_product_ids_for_brand(store_a_products, store_b_products, brand):
    """Get matching product IDs from LLM and build full match objects locally."""
    # Create maps so we can get the API to output only product_ids (saving output tokens), then quickly look up the corresponding product details
    store_a_map = {p['product_id']: p for p in store_a_products}
    store_b_map = {p['sku']: p for p in store_b_products}
    
    for attempt in range(3):  # Max 3 retries
        try:
            client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY"),
            )

            prompt = f"""" You are a diligent merchandiser working at a grocery store chain with strong attention to detail.

            store_a_products = {store_a_products}
            store_b_products = {store_b_products}

            Inference which items in these two lists are the same item.

            To be the same item, the two products must satisfy the following:
            (1) Be of the same brand\n
            (2) Be of the same type and flavor (e.g. flavor = pineapple, type - coffee cake).

            Return ONLY a JSON array of matching ID pairs:
                [
                    {{"product_a_id": "PRODUCT_ID_FROM_STORE_A", "product_b_id": "SKU_FROM_STORE_B"}}
                ]

            Return empty array [] if no matches."""
            
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=[types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )],
                config=types.GenerateContentConfig(
                    temperature=1,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192,
                )
            )
            
            response_text = response.candidates[0].content.parts[0].text
            print(f'API call for {brand} complete. Response length: {len(response_text)} chars')
            
            # Extract and parse JSON
            matches = []
            matching_ids = extract_json_from_text(response_text)
            
            # Build full match objects from IDs
            for pair in matching_ids:
                product_a_id = pair.get('product_a_id')
                product_b_id = pair.get('product_b_id')
                
                if product_a_id in store_a_map and product_b_id in store_b_map:
                    product_a = store_a_map[product_a_id]
                    product_b = store_b_map[product_b_id]
                    
                    # Calculate price differences
                    price_a = float(product_a.get('price', 0))
                    price_b = float(product_b.get('price', 0))
                    store_b_price_diff_vs_store_a = price_b - price_a
                    store_b_price_diff_vs_store_a_percent = (store_b_price_diff_vs_store_a / price_a * 100) if price_a else 0
                    
                    # Format differences directly when creating the match
                    match = {
                        'product_a': product_a,
                        'product_b': product_b,
                        'price_a': price_a,
                        'price_b': price_b,
                        'price_diff': format_price_diff(store_b_price_diff_vs_store_a),
                        'price_diff_percent': format_price_diff(store_b_price_diff_vs_store_a_percent, is_percentage=True)
                    }
                    matches.append(match)
            
            print(f"Found {len(matches)} matches for {brand}")
            return matches
            
        except (ClientError, ServerError, APIError) as e:
            wait_time = 4 * (2 ** attempt)  # Simple exponential backoff
            print(f'API error on attempt {attempt+1}/3: {e}')
            
            if attempt < 2:
                print(f'Retrying in {wait_time}s...')
                await asyncio.sleep(wait_time)
            else:
                print(f"Failed after 3 attempts")
        
        except Exception as e:
            print(f'Unexpected error: {e}')
            break
            
    return []  # Return empty list if all retries failed

def filter_matches(matches):
    """Filter out matches with size or quantity mismatches."""
    filtered_matches = []
    removed_count = 0
    
    for match in matches:
        product_a = match['product_a']
        product_b = match['product_b']
        
        # Check for size mismatch
        if product_a.get('size') and product_b.get('size'):
            size_a = extract_numeric_value(product_a['size'])
            size_b = extract_numeric_value(product_b['size'])
            
            if size_a and size_b and size_a != size_b:
                removed_count += 1
                continue  # Skip this match
        
        # Check for quantity mismatch
        if product_a.get('quantity') and product_b.get('quantity'):
            try:
                qty_a = int(product_a['quantity'])
                qty_b = int(product_b['quantity'])
                
                if qty_a != qty_b:
                    removed_count += 1
                    continue  # Skip this match
            except (ValueError, TypeError):
                pass  # If we can't parse quantities, don't filter
        
        # Passed all filters
        filtered_matches.append(match)
    
    print(f"Filtered out {removed_count} matches, keeping {len(filtered_matches)}")
    return filtered_matches

async def find_all_matching_products_and_compare_prices(file_path_a, file_path_b, output_path):
    """Main function to find and compare matching products."""
    all_matches = []
    coroutines = []
    
    # Track start time for performance reporting
    start_time = datetime.now()
    
    try:
        # Group products by brand
        brand_products = get_overlapping_brands_and_their_products(file_path_a, file_path_b)
        print(f'Grouping products took {(datetime.now() - start_time).seconds} seconds')
        
        # Google API has rate limit of 15 requests / min, lowering to 13 to add some buffer since there are a lot of brands and I want this to execute in its entirety safely
        rate_limiter = LeakyBucketLimiter(rate=12/60, capacity=12)
        
        # Create a task for each brand (limit to 10 for testing)
        counter = 0
        product_count = 0
        
        for brand, product_data in brand_products.items():
            # if counter >= 50:  # Comment this out for full processing
            #     break
                
            store_a_count = len(product_data['store_a_products'])
            store_b_count = len(product_data['store_b_products'])
            product_count += store_a_count + store_b_count
            
            print(f'Processing {brand}: {store_a_count} products in store A, {store_b_count} in store B')
            coroutines.append(find_matching_product_ids_for_brand(
                product_data['store_a_products'], 
                product_data['store_b_products'],
                brand
            ))
            counter += 1

        # Execute API calls concurrently with rate limiting
        api_start = datetime.now()
        
        api_responses = await asyncio.gather(*map(rate_limiter.wrap, coroutines))
        print(f'API calls took {(datetime.now() - api_start).seconds} seconds')
        
        for response in api_responses:
            if response:  # Only add non-empty responses
                all_matches.extend(response)
        
        # Save unfiltered matches for reference
        save_results(all_matches, output_path.replace('.json', '_no_cleanup.json'), 
                    "Saved raw matches")
            
        # Filter out product matches where the size or quantity is mismatched
        filtered_matches = filter_matches(all_matches)

        # Sort product matches based on magnitude of price difference in dollars
        filtered_matches.sort(key=lambda m: abs(float(m['price_diff'].replace('$', '').replace('+', ''))), reverse=True)
        
        save_results(filtered_matches, output_path)

        total_time = (datetime.now() - start_time).seconds
        print(f"Completed in {total_time} seconds. Found {len(filtered_matches)} matches across {counter} brands.")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        
        if all_matches:
            save_results(all_matches, output_path.replace('.json', '_error.json'),
                        "Saved results before error")

if __name__ == "__main__":
    file_path_a = r"C:\Users\Susie's PC\better-basket-technical-assessment\grocery_store_a_relevant_fields.json"
    file_path_b = r"C:\Users\Susie's PC\better-basket-technical-assessment\grocery_store_b_relevant_fields.json"
    output_path = r"C:\Users\Susie's PC\better-basket-technical-assessment\match_results.json"
    
    try:
        asyncio.run(find_all_matching_products_and_compare_prices(file_path_a, file_path_b, output_path))
    except KeyboardInterrupt:
        print("Program terminated by user.")