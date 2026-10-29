# import pandas as pd

# data = {
#     'Order_ID': [1, 2, 3],
#     'Customer': ['Ali', 'Sara', 'Ahmed'],
#     'City': ['Karachi', 'Lahore', 'Karachi'],
#     'Products': ['Laptop, Mouse','Keyboard, Mouse','Laptop, Keyboard, Mouse'],
#     'Quantities': ['1, 2', '1, 1', '1, 1, 2'],
#     'Prices': ['80000, 1500', '3000, 1500', '80000, 3000, 1500']
# }

# df = pd.DataFrame(data)

# print("Original DataFrame:")
# print(df)

# print("\nColumns containing multiple values:")
# print("Products, Quantities, Prices")

# normalized_data = []

# for _, row in df.iterrows():
#     products = row['Products'].split(', ')
#     quantities = row['Quantities'].split(', ')
#     prices = row['Prices'].split(', ')

#     for product, quantity, price in zip(
#         products, quantities, prices
#     ):
#         normalized_data.append({
#             'Order_ID': row['Order_ID'],
#             'Customer': row['Customer'],
#             'City': row['City'],
#             'Product': product,
#             'Quantity': int(quantity),
#             'Price': int(price)
#         })

# normalized_df = pd.DataFrame(normalized_data)

# print("\nNormalized DataFrame:")
# print(normalized_df)

# print("\nRows before normalization:", len(df))
# print("Rows after normalization:", len(normalized_df))

import pandas as pd

data = {
    'Order_ID': [1, 2, 3],
    'Customer': ['Ali', 'Sara', 'Ahmed'],
    'City': ['Karachi', 'Lahore', 'Karachi'],
    'Products': ['Laptop, Mouse', 'Keyboard, Mouse', 'Laptop, Keyboard, Mouse'],
    'Quantities': ['1, 2', '1, 1', '1, 1, 2'],
    'Prices': ['80000, 1500', '3000, 1500', '80000, 3000, 1500']
}
df = pd.DataFrame(data)

print("Orignal Data:\n")
print(df)

df['Products'] = df['Products'].str.split(', ')
df['Quantities'] = df['Quantities'].str.split(', ')
df['Prices'] = df['Prices'].str.split(', ')

df_normalized = df.explode(['Products', 'Quantities', 'Prices']).reset_index(drop=True)

df_normalized['Quantities'] = df_normalized['Quantities'].astype(int)
df_normalized['Prices'] = df_normalized['Prices'].astype(int)


print("\nNormalized Data:\n")
print(df_normalized)

