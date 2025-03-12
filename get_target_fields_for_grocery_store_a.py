import json
import re

def extract_product_info(file_path):
    """
    Extract product name and price from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries with product name and price
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    products = []
    
    # Initialize counters for field statistics
    total_items = 0
    field_counts = {
        "product_name": 0,
        "brand": 0,
        "product_id": 0,
        "price": 0,
        "size": 0,
        "quantity": 0
    }
    
    for item in data:
        if 'data' in item and 'product' in item['data']:
            total_items += 1
            product = item['data']['product']
            
            price = None
            size = None
            quantity = None

            name = product.get('name')
            if name:
                field_counts["product_name"] += 1
                
            brand = product.get('brand')
            if brand:
                field_counts["brand"] += 1
                
            product_id = product.get('id')
            if product_id:
                field_counts["product_id"] += 1
                
            if brand is None:
                brand = "N/A"
            else:
                brand = brand.strip("'").upper()

            short_description = product.get('shortDescription')
            size_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:oz|fl\s*oz|fluid\s*oz|fluid\s*ounce|ounce|gallon|ml|l|g|kg|lb|lbs|square feet)\b', name, re.IGNORECASE)
            # Search sort description if size not in title
            if not size_match:
                size_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:oz|fl\s*oz|fluid\s*oz|fluid\s*ounce|ounce|gallon|ml|l|g|kg|lb|lbs|square feet)\b', short_description, re.IGNORECASE)
            
            # Normalize units
            if size_match:
                size = size_match.group(0).lower()
                # Normalize units
                size = size.replace('ounce', 'oz') \
                        .replace('pound', 'lb') \
                        .replace('liter', 'l') \
                        .replace('gram', 'g')
                field_counts["size"] += 1
            else:
                # Check for descriptive sizes
                size_desc_match = re.search(r'\b(mini|small|medium|large|x-large)\b', name, re.IGNORECASE)
                if size_desc_match:
                    size = size_desc_match.group(0).lower()
                    field_counts["size"] += 1
            
            # Extract quantity
            quantity_match = re.search(r'\b(\d+)\s*(?:count|ct|pk|pc|piece|roll|pack)s?\b', name, re.IGNORECASE)
            if not size_match and not quantity_match:
                quantity_match = re.search(r'\b(\d+)\b', name, re.IGNORECASE)
            if quantity_match:
                count = quantity_match.group(1)
                quantity = count
                field_counts["quantity"] += 1
            
            # Extract price - handle nested dictionaries safely
            price_info = product.get('priceInfo')
            if price_info is not None:
                current_price = price_info.get('currentPrice')
                if current_price is not None:
                    price = current_price.get('price')
                    if price is not None:
                        field_counts["price"] += 1
            
            if name and price is not None:
                products.append({
                    "product_name": name,
                    "brand": brand,
                    "product_id": product_id,
                    "price": price,
                    "size": size,
                    "quantity": quantity
                })
    
    # Print statistics summary
    print("\n" + "="*50)
    print(" PRODUCT DATA EXTRACTION SUMMARY ")
    print("="*50)
    print(f"\nTotal items processed: {total_items}")
    print(f"Items with valid product name and price (extracted): {len(products)}")
    print("\nField population statistics:")
    
    for field, count in field_counts.items():
        percentage = (count / total_items) * 100 if total_items > 0 else 0
        print(f"  - {field:<12}: {count:>5} / {total_items} ({percentage:.1f}%)")
    
    print("\nDiscarded items: {0}".format(total_items - len(products)))
    print("="*50)
    
    return products

# Usage
products = extract_product_info(r"C:\Users\Susie's PC\better-basket-technical-assessment\grocery_store_a.json")

# Output as JSON
output_json = json.dumps(products, indent=2)

# Optionally save to a file
with open(r"C:\Users\Susie's PC\better-basket-technical-assessment\grocery_store_a_relevant_fields.json", 'w', encoding='utf-8') as f:
    f.write(output_json)