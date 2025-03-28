import requests

def get_alternative_products(categories):
    alternatives = []

    for category in categories:
        url = f"https://world.openfoodfacts.org/api/v2/search?categories_tags={category}&fields=product_name,nutriscore_grade,nutriments,image_url,positive_points,ingredients_text"
        response = requests.get(url)

        if response.status_code == 200:
            products = response.json().get('products', [])

            for product in products:
                product_name = product.get('product_name', 'Unknown')
                nutriscore = product.get('nutriscore_grade', 'e')
                product_image = product.get('image_url', 'static/default.jpg')
                nutriments = product.get('nutriments', {})
                ingredients_text = product.get('ingredients_text', 'No ingredient info')

                # Extracting Positive Points
                proteins = nutriments.get('proteins_value', 0)
                fiber = nutriments.get('fiber_value', 0)
                fruits_veggies = nutriments.get('fruits_vegetables_legumes_value', 0)

                positive_points = {
                    "Proteins": f"{proteins}g",
                    "Fiber": f"{fiber}g"
                }

                # Warning if fruits/veggies info is missing
                warning = ""
                if fruits_veggies == 0:
                    warning = " Warning: Fruit/vegetable content is missing from label"

                if nutriscore in ['a', 'b']:
                    alternatives.append({
                        "name": product_name,
                        "nutriscore": nutriscore.upper(),
                        "image": product_image,
                        "positive_points": positive_points,
                        "warning": warning
                    })

    return alternatives[:6]
