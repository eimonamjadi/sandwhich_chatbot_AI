# Write the Streamlit app to app.py

# Import required packaged
import streamlit as st
import openai
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
#import os
import json

st.title("Subway Sandwich Customization Chatbot")

# Add API key input at the top of the app
api_key = st.text_input("Enter your OpenAI API key:", type="password") # use this for the video and paste inside a key
if not api_key:
    st.warning("Please enter your OpenAI API key to enable order summaries.")
else:
    openai.api_key = api_key

def generate_summary(user_name: str, order_details: list, check_combo: str) -> str:
    """
    Generate a formatted order summary using OpenAI's GPT model.
    """
    try:
        # Validate input, ensuring we have the required parameters
        if not user_name or not order_details:
            raise ValueError("User name and order details are required")

        # Initialize the item string and other values
        items_str = "" # will hold the summary format
        i = 0 # keeps track of the entry/order we're on
        subtotal = 0 # will keep track of the price without tax
        for item in order_details:
          # the order will either have a drink or won't due to the our combo format
          if 'drink' in item:
              i = i + 1 # increase the count
              sand_price = item["line_total"] # get the price of the sandwhich
              # note that we do not add extra pricing for the combo, since it will be taken into account for line_total later
              subtotal = subtotal + sand_price # combine prices
              # Organize the attributes of each order into the following format:
              items_str += (
                   f"{i}. {item['name']}, "
                   f"Size: {item['size']}, "
                   f"Bread: {item['bread']}, "
                   f"Cheese: {item['cheese']}, "
                   f"Veggies: {', '.join(item['veggies'])}, "
                   f"Sauces: {', '.join(item['sauces'])}, "
                   f"Drink: {item['drink']}, "
                   f"Side: {item['side']}, "
                   f"Quantity: {item['quantity']}, "
                   f"Line Price: ${item['line_total']:.2f}\n"
                    )
          else:
              # similar to before but without combo items (everything except drink + side )
              i = i + 1
              subtotal = subtotal + item["line_total"]
              items_str += (
                  f"{i}. {item['name']}, "
                  f"Size: {item['size']}, "
                  f"Bread: {item['bread']}, "
                  f"Cheese: {item['cheese']}, "
                  f"Veggies: {', '.join(item['veggies'])}, "
                  f"Sauces: {', '.join(item['sauces'])}, "
                  f"Quantity: {item['quantity']}, "
                  f"Line Price: ${item['line_total']:.2f}\n"
                  )
        tax = round(subtotal * 0.08, 2) # find the tax
        total = round(subtotal + tax, 2) # apply the tax to the total
        # If no API key, return basic summary
        if not api_key:
            return format_basic_summary(user_name, order_details, subtotal, tax, total)

        # Create message for ChatGPT
        messages = [
            {"role": "system", "content": "You are a helpful assistant that formats order summaries."},
            {"role": "user", "content": f"""
            Please format this order summary:

            Customer Name: {user_name}

            Orders:
            {items_str}

            Subtotal: \${subtotal:.2f}
            Tax (8\%): \${tax:.2f}
            Total: \${total:.2f}
            """}
        ]

        # Make API call to OpenAI
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message['content'].strip()

        except Exception as e:
            st.error(f"Error generating summary with OpenAI: {str(e)}")
            return format_basic_summary(user_name, order_details, subtotal, tax, total)

    except KeyError as e:
        raise ValueError(f"Missing required field in order details: {str(e)}")

# Ignore this function, not used if API works
def format_basic_summary(user_name, order_details, subtotal, tax, total):
    """Fallback function to format summary without OpenAI"""
    summary = f"""
    Order Summary:
    Customer Name: {user_name}

    Orders:
    """
    for i, item in enumerate(order_details, start=1):
        summary += f"""
        {i}. {item['name']}
           Size: {item['size']}
           Bread: {item['bread']}
           Cheese: {item['cheese']}
           Veggies: {', '.join(item['veggies'])}
           Sauces: {', '.join(item['sauces'])}
           Quantity: {item['quantity']}
           Line Price: ${item['line_total']:.2f}
        """

    summary += f"""
    Subtotal: \${subtotal:.2f}
    Tax (8\%): \${tax:.2f}
    Total: \${total:.2f}

    Thank you for your order!
    """
    return summary

#### RAG functions and documents
# Our current menu at subway, with ingredients
corpus = """
Rotisserie-Style Chicken: Rotisserie-Style Chicken, Footlong,  Multigrain, American, Lettuce, Tomatoes, Red Onions.
Carved Turkey: Turkey Breast, Bacon, Footlong, Italian, American, Spinach, Tomatoes, Red Onions.
Roast Beef: Angus Roast Beef, Footlong, Multigrain Bread, American, Lettuce, Tomatoes, Red Onions.
Chicken & Bacon Ranch Melt:  Rotisserie-Style Chicken, Bacon, Footlong, Italian, Monterey Cheddar, Lettuce, Tomatoes, Red Onions, Peppercorn Ranch.
Steak & Cheese: Steak, Footlong, Italian, American, Green Peppers, Red Onions.
Sweet Onion Chicken Teriyaki: Chicken Breast, Footlong, Multigrain, American, Lettuce, Tomatoes, Red Onions, Sweet Onion Teriyaki Sauce.
Subway Club: Black Forest Ham, Footlong, Multigrain, American, Lettuce, Tomatoes, Red Onions, Mayo.
Oven Roasted Chicken: Chicken Breast, Footlong, Multigrain, American, Lettuce, Tomatoes, Red Onions.
Italian BMT: Genoa Salami, Spicy Pepperoni, Black Forest Ham, Footlong, Italian, Provolone, Lettuce, Tomatoes, Red Onions, Parmesan Vinaigrette.
Tuna: Tuna, Footlong, Italian, Lettuce, Tomatoes, Red Onions, Mayo.
Turkey Breast: Turkey Breast, Footlong, Italian, Provolone, Lettuce, Tomatoes, Green Peppers, Cucumbers, Mayo.
Black Forest Ham: Black Forest Ham, Footlong, Italian, Swiss, Lettuce, Tomatoes, Green Peppers, Cucumbers, Mustard.
Veggie Delite: Footlong, Multigrain, Lettuce, Tomatoes, Green Peppers, Cucumbers and Onions, Oil & Vinegar.
Cold Cut Combo: Turkey Bologna, Salami, Black Forest Ham, Footlong, Italian, Provolone, Lettuce, Tomatoes, Mayo.
Meatball Marinara: Meatballs, Footlong, Italian, Swiss.
Spicy Italian: Spicy Pepperoni, Salami, Footlong, Italian, Provolone, Lettuce, Tomatoes, Cucumber, Mayo.
Rotisserie-Style Chicken: Rotisserie-Style Chicken, 6-inch,  Multigrain, American, Lettuce, Tomatoes, Red Onions.
Carved Turkey: Turkey Breast, Bacon, 6-inch, Italian, American, Spinach, Tomatoes, Red Onions.
Roast Beef: Angus Roast Beef, 6-inch, Multigrain Bread, American, Lettuce, Tomatoes, Red Onions.
Chicken & Bacon Ranch Melt:  Rotisserie-Style Chicken, Bacon, 6-inch, Italian, Monterey Cheddar, Lettuce, Tomatoes, Red Onions, Peppercorn Ranch.
Steak & Cheese: Steak, 6-inch, Italian, American, Green Peppers, Red Onions.
Sweet Onion Chicken Teriyaki: Chicken Breast, 6-inch, Multigrain, American, Lettuce, Tomatoes, Red Onions, Sweet Onion Teriyaki Sauce.
Subway Club: Black Forest Ham, 6-inch, Multigrain, American, Lettuce, Tomatoes, Red Onions, Mayo.
Oven Roasted Chicken: Chicken Breast, 6-inch, Multigrain, American, Lettuce, Tomatoes, Red Onions.
Italian BMT: Genoa Salami, Spicy Pepperoni, Black Forest Ham, 6-inch, Italian, Provolone, Lettuce, Tomatoes, Red Onions, Parmesan Vinaigrette.
Tuna: Tuna, 6-inch, Italian, Lettuce, Tomatoes, Red Onions, Mayo.
Turkey Breast: Turkey Breast, 6-inch, Italian, Provolone, Lettuce, Tomatoes, Green Peppers, Cucumbers, Mayo.
Black Forest Ham: Black Forest Ham, 6-inch, Italian, Swiss, Lettuce, Tomatoes, Green Peppers, Cucumbers, Mustard.
Veggie Delite: 6-inch, Multigrain, Lettuce, Tomatoes, Green Peppers, Cucumbers and Onions, Oil & Vinegar.
Cold Cut Combo: Turkey Bologna, Salami, Black Forest Ham, 6-inch, Italian, Provolone, Lettuce, Tomatoes, Mayo.
Meatball Marinara: Meatballs, 6-inch, Italian, Swiss.
Spicy Italian: Spicy Pepperoni, Salami, 6-inch, Italian, Provolone, Lettuce, Tomatoes, Cucumber, Mayo
"""
document = corpus.replace('\n','').split('.') # split to create a list of documents
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') # ready up the tokenizer and model
model_tokenize = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# embed the doc
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():# disable gradients
        vectors = model_tokenize(**inputs).pooler_output

    # Return the vector embeddings of the input text. This vector captures the semantic meaning of the input text in a high-dimensional space.
    return vectors

document_embeddings = torch.vstack([embed_text(doc) for doc in document])
### check how to implement this into summary
def get_completion_rag(query, doc=document, document_embed=document_embeddings, model="gpt-3.5-turbo"):
  # first look for the index of the similar doc
    query_embedding = embed_text(query)
    similarities = cosine_similarity(query_embedding, document_embed) # continue using
    # cosine sim since we are using a pretrained model to embed the docs

    most_relevant_doc_index = similarities.argmax()
    # then create a response for the bot to use
    prompt = f"Based on the following information deliminated in <>, Based on the input deliminated in ''', recommend a different item. Make sure to share detials on the ingredients. Make sure it is either 6-inches or a Footlong and that it is in the information: \nCustomer:'''{query}''' \nInformation: <{doc[most_relevant_doc_index]}>"
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]
def format_sandwich(query):
    # Take the query string input and create a dictionary based on the composition of the order
    input = f"""
    The Subway order is delimited by <> below.
    Your task is to identify the ingredients and the size of this Subways \
    sancwich order. Make sure to check if it contains the following:
    1. Sandwich classification will only one element from the following list: [Italian B.M.T., Turkey Breast, Tuna, Meatball Marinara, Spicy Italian, Veggie Delite, Steak & Cheese, Chicken & Bacon Ranch, Black Forest Ham, Subway Club, Roast Beef, Sweet Onion Chicken Teriyaki, Oven Roasted Chicken, Turkey Italiano Melt, Buffalo Chicken, Cold Cut Combo]
    2. Size must be either in [Footlong, 6-inches]
    3. Bread must be in [Italian, Wheat, Multigrain, Italian Herbs & Cheese, Gluten-free]
    4. Cheese must be in [American, Provolone, Swiss, No Cheese]
    5. Vegetables can be more than one value in [Lettuce, Tomato, Onions, Peppers, Olives, Cucumbers, Spinach, Jalapeños, Pickles]
    6. Sauces can be more than one value in [Mayo, Mustard, Chipotle Southwest, Sweet Onion, Oil & Vinegar, No Sauce]

    If it does, then return the values found in the form of a dictionary.
    Where the keys are [Sandwich, Size, Bread, Cheese, Veggies, Sauces] \
    and their proper values are what you identified.

    Subway Order: <{query}>

    """
    messages = [{"role": "user", "content": input}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]
#### End of RAG

# Title Before the Api code
st.title("Subway Sandwich Customization Chatbot")

# Sandwich dictionary
# format {Sandwhich: {Size1: (calories pair), Size2: (calories Pair), price_size1: price, price_size2: price}}
sandwiches = {
    "Italian B.M.T.": {
        "6-inch": (410, 450),
        "Footlong": (820, 900),
        "price_6": 5.00,
        "price_footlong": 8.50
    },
    "Turkey Breast": {
        "6-inch": (280, 320),
        "Footlong": (560, 640),
        "price_6": 5.00,
        "price_footlong": 8.50
    },
    "Tuna": {
        "6-inch": (480, 530),
        "Footlong": (960, 1060),
        "price_6": 5.50,
        "price_footlong": 9.00
    },
    "Meatball Marinara": {
        "6-inch": (460, 500),
        "Footlong": (920, 1000),
        "price_6": 5.00,
        "price_footlong": 8.50
    },
    "Spicy Italian": {
        "6-inch": (450, 500),
        "Footlong": (900, 1000),
        "price_6": 5.00,
        "price_footlong": 8.50
    },
    "Veggie Delite": {
        "6-inch": (230, 270),
        "Footlong": (460, 540),
        "price_6": 4.50,
        "price_footlong": 7.50
    },
    "Steak & Cheese": {
        "6-inch": (340, 380),
        "Footlong": (680, 760),
        "price_6": 6.00,
        "price_footlong": 9.50
    },
    "Chicken & Bacon Ranch": {
        "6-inch": (570, 620),
        "Footlong": (1140, 1240),
        "price_6": 6.50,
        "price_footlong": 10.50
    },
    "Black Forest Ham": {
        "6-inch": (290, 330),
        "Footlong": (580, 660),
        "price_6": 5.00,
        "price_footlong": 8.50
    },
    "Subway Club": {
        "6-inch": (310, 360),
        "Footlong": (620, 720),
        "price_6": 5.50,
        "price_footlong": 9.00
    },
    "Roast Beef": {
        "6-inch": (320, 360),
        "Footlong": (640, 720),
        "price_6": 5.50,
        "price_footlong": 9.00
    },
    "Sweet Onion Chicken Teriyaki": {
        "6-inch": (370, 420),
        "Footlong": (740, 840),
        "price_6": 6.00,
        "price_footlong": 9.50
    },
    "Oven Roasted Chicken": {
        "6-inch": (270, 310),
        "Footlong": (540, 620),
        "price_6": 5.00,
        "price_footlong": 8.50
    },
    "Turkey Italiano Melt": {
        "6-inch": (450, 500),
        "Footlong": (900, 1000),
        "price_6": 5.50,
        "price_footlong": 9.00
    },
    "Buffalo Chicken": {
        "6-inch": (350, 400),
        "Footlong": (700, 800),
        "price_6": 6.00,
        "price_footlong": 9.50
    },
    "Cold Cut Combo": {
        "6-inch": (310, 350),
        "Footlong": (620, 700),
        "price_6": 4.50,
        "price_footlong": 7.50
    }
}
# description for sandwiches
description = {
    "Cold Cut Combo": "Step into a slice of SUBWAY® tradition with our Cold Cut Classic! This time-honored favorite layers thinly sliced turkey bologna, turkey ham, and turkey salami, each offering its unique twist on classic deli flavors.",
    "Italian B.M.T.": "Dive into the robust flavors of our Italian BMT, a tribute to the timeless allure of Italian deli classics! This sandwich is stacked with slices of zesty Genoa salami, spicy pepperoni, and rich Black Forest ham, each bringing its own unique flavor to the table.",
    "Turkey Breast": "Savor the irresistible charm of our Turkey Temptation! This sandwich is a healthful delight, featuring succulent slices of smoked turkey that are both tender and tantalizing.",
    "Tuna": "Tuna and creamy mayonnaise is lovingly blended together to make one of the world's favorite comfort foods.",
    "Meatball Marinara": "Indulge in the rich, robust flavors of our Meatball Marinara! This sandwich is a hearty homage to Italian culinary traditions, featuring a generous helping of Italian-style meatballs.",
    "Spicy Italian": "Ignite your palate with our Spicy Italian Sizzler! This sandwich is a fiery feast, crafted for those who dare to add a little heat to their meal. It features a bold and spicy ensemble of pepperoni and salami.",
    "Veggie Delite": "Embark on a culinary journey with our Vibrant Veggie Delite! This sandwich is a celebration of freshness, featuring a crisp and crunchy assortment of garden delights.",
    "Steak & Cheese": "Dive into the savory depths of our Sizzling Steak Sensation! Each bite is packed with mouth-watering slices of lean, tender steak, expertly seasoned and grilled to perfection.",
    "Chicken & Bacon Ranch": "Savor the irresistible charm of our Chicken & Bacon Ranch Melt! This delectable sandwich features succulent chicken strips, tender and juicy, layered under strips of premium, crispy bacon that sizzle with flavor.",
    "Black Forest Ham": "The Black Forest Ham has never been better. Load it up with all the veggies you like on your choice of freshly baked bread. Even try it fresh toasted with melty cheese and mustard. Yum!",
    "Subway Club": "Embrace the delightful fusion of flavors in our Ultimate Meat Medley! This sandwich is a carnivore’s dream, featuring a perfect combination of thinly sliced turkey breast, tender roast beef, and succulent ham.",
    "Roast Beef": "This tasty number, with less than 6g of fat, is piled high with lean roast beef and your choice of fresh veggies.",
    "Sweet Onion Chicken Teriyaki": "Experience the allure of our Teriyaki Temptation, where each bite is a journey to flavor paradise! This sandwich features succulent chicken strips, marinated in a rich teriyaki glaze that balances sweetness with a savory punch.",
    "Oven Roasted Chicken": "Feast your senses on our Oven Roasted Chicken! This sumptuous sandwich starts with tender, oven-roasted chicken, seasoned to perfection and piled high on freshly baked bread that's soft on the inside and crusty on the outside.",
    "Turkey Italiano Melt": "Imagine freshly baked bread stuffed with tender sliced turkey breast, ham, crispy bacon, melted cheese, and your choice of tasty vegetables and condiments. Now, stop imagining.",
    "Buffalo Chicken": "Chicken breast strips marinated in spicy buffalo sauce. Try it"
}
### Beginning of the user interface creation
username = st.text_input("Hello! What's your name?")
if username:
    st.write(f"Nice to meet you, {username}! Let's select a sandwich.")
    # initialize global variables that need to be appended, updated
    # or will require a boolean conditional statement later in the code
    if "selected_sandwiches" not in st.session_state:
        st.session_state.selected_sandwiches = []
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "current_sandwich" not in st.session_state:
        st.session_state.current_sandwich = None
    if "combo" not in st.session_state:
        st.session_state.combo = False
    if "order_complete" not in st.session_state:
        st.session_state.order_complete = False
    if "append" not in st.session_state:
        st.session_state.append = False
    if "order_details" not in st.session_state:
        st.session_state.order_details = []

    # Display sandwich list if we haven't chosen one yet
    if st.session_state.current_step == 0: # Beginning of the steps
        st.write("Please select a sandwich:")
        for s in sandwiches.keys(): # for every sandwich we sell
            if st.button(s): # create a button that if pressed
                st.session_state.current_sandwich = s # append the chosen value to the global variable
                st.session_state.current_step = 1 # go to the next step
            st.write(description[s])

    # Step 1: Choose Size
    if st.session_state.current_step == 1 and st.session_state.current_sandwich: # if they chose a sandwich
        st.write(f"You selected: {st.session_state.current_sandwich}")
        st.write("Step 1: Choose Size")
        col1, col2 = st.columns(2) # create two columns to use two different buttons
        with col1:
            if st.button("6-inch"):
                st.session_state.size = "6-inch"
                st.session_state.current_step = 2
        with col2:
            if st.button("Footlong"):
                st.session_state.size = "Footlong"
                st.session_state.current_step = 2

    # Step 2: Choose Bread
    if st.session_state.current_step == 2: # repeat many of the same structure as before
        st.write("Step 2: Choose Bread Type")
        bread_types = ["Italian", "Wheat", "Multigrain", "Italian Herbs & Cheese", "Gluten-free"]
        for b in bread_types:
            if st.button(b):
                st.session_state.bread = b
                st.session_state.current_step = 3
    # Step 3: Choose Cheese
    if st.session_state.current_step == 3:
        st.write("Step 3: Choose Cheese")
        cheese_types = ["American", "Provolone", "Swiss", "No Cheese"]
        for c in cheese_types:
            if st.button(c):
                st.session_state.cheese = c
                st.session_state.current_step = 4

    # Step 4: Choose Veggies
    if st.session_state.current_step == 4: # Here we want to allow for multiple valid choices, not just one
        st.write("Step 4: Choose Veggies")
        veggies = ["Lettuce", "Tomato", "Onions", "Peppers", "Olives", "Cucumbers", "Spinach", "Jalapeños", "Pickles"]
        selected_veggies = st.multiselect("Select your veggies:", veggies) # MultiSelect allows us to create a list of chosen values!
        if st.button("Confirm Veggies"): # Once they finished their selection, they can press this to continue
            st.session_state.veggies = selected_veggies
            st.session_state.current_step = 5

    # Step 5: Choose Sauces
    if st.session_state.current_step == 5: # Same as step 5
        st.write("Step 5: Choose Sauces/Condiments")
        sauces = ["Mayo", "Mustard", "Chipotle Southwest", "Sweet Onion", "Oil & Vinegar", "No Sauce"]
        selected_sauces = st.multiselect("Select your sauces:", sauces)
        if st.button("Confirm Sauces"):
            st.session_state.sauces = selected_sauces
            st.session_state.current_step = 6

    # Step 6: Make it a combo
    if st.session_state.current_step == 6:
      st.write("Step 6: Make it a Combo")
      drink_choices = ["Coca Cola", "Coca Cola Zero Sugar", "Water", "Powerade", "Dr. Pepper", "Lemonade", "Sprite", "Fanta"]
      sides_choices = ["Chocolate Chip Cookie", "Sugar Cookie", "Lays Original", "Spicy Dill Pickle Chips", "Cheetos", "Flamin' Hot Cheetos", "Barbecue Lays", "Jalapeño Kettle Chips"]
      col3, col4 = st.columns(2)
      with col3:
          # Combo
          if st.button("Make a combo"):
              st.session_state.current_step = 7 # Send to combo customization
      with col4:
          # No combo
          if st.button("No Combo"):
              st.session_state.drink = "No Drink"
              st.session_state.side = "No Side"
              st.session_state.current_step = 8 # Send to Quantity and Review
    # Step 7: Customize combo
    if st.session_state.current_step == 7:
        drink_choices = ["Coca Cola", "Coca Cola Zero Sugar", "Water", "Powerade", "Dr. Pepper", "Lemonade", "Sprite", "Fanta"]
        sides_choices = ["Chocolate Chip Cookie", "Sugar Cookie", "Lays Original", "Spicy Dill Pickle Chips", "Cheetos", "Flamin' Hot Cheetos", "Barbecue Lays", "Jalapeño Kettle Chips"]
        #
        selected_drink = st.selectbox("Choose a drink:", drink_choices)
        selected_side = st.selectbox("Choose a side:", sides_choices)
        if st.button("Confirm Combo"):
            st.session_state.combo = True
            st.session_state.drink = selected_drink
            st.session_state.side = selected_side
            st.session_state.current_step = 8

    # Step 8: Confirm Quantity and Review Order
    if st.session_state.current_step == 8:
        # Here we want the customer to see if they want to order more of the same item/combo before adding to cart
        st.write("Step 6: Confirm Quantity and Review Order")
        quantity = st.selectbox("How many of this sandwich?", [1,2,3,4,5]) # select only one choice
        st.session_state.quantity = quantity
        if st.button("Add to Order"):
            st.session_state.current_step = 9

    if st.session_state.current_step == 9:
        # After the order is added, we want to calculate the price, and create a order_receipt/ append the order_details
        # after retreiving and calculating the line total
        quantity = st.session_state.quantity # the number the user selected before
        # sand_price finds the price depending on the sandwich they chose
        sand_price = sandwiches[st.session_state.current_sandwich]["price_6"] if st.session_state.size == "6-inch" else sandwiches[st.session_state.current_sandwich]["price_footlong"]
        if st.session_state.combo == True and st.session_state.append == False: # we have this append value to avoid duplication of the order when this step runs
            price = sand_price + 4  # note that depending on whether it was a combo or not, we add an extra $4 to the base price
            line_total = price * quantity # find the line's subtotal as it is untaxed
            st.session_state.order_details.append({
                "name": st.session_state.current_sandwich,
                "size": st.session_state.size,
                "bread": st.session_state.bread,
                "cheese": st.session_state.cheese,
                "veggies": st.session_state.veggies,
                "sauces": st.session_state.sauces,
                "drink": st.session_state.drink,
                "side": st.session_state.side,
                "quantity": quantity,
                "line_total": line_total
                  })
            st.session_state.append = True # set to true to avoid duplication
        elif st.session_state.combo == False and st.session_state.append == False:
            # same but without the extra cash since it is not a combo
            # note that this one does not include 'drink' nor 'side'
            price = sand_price
            line_total = price * quantity
            st.session_state.order_details.append({
                "name": st.session_state.current_sandwich,
                "size": st.session_state.size,
                "bread": st.session_state.bread,
                "cheese": st.session_state.cheese,
                "veggies": st.session_state.veggies,
                "sauces": st.session_state.sauces,
                "quantity": quantity,
                "line_total": line_total
                  })
            st.session_state.append = True
        st.write("Order updated! Would you like to order more?")
        col5, col6 = st.columns(2) # two choices
        with col5:
            if st.button("Yes"): # Will restart to the very first step where they pick a new sandwich/meat
                st.session_state.current_sandwich = None
                st.session_state.current_step = 0
                st.session_state.combo = False
                st.session_state.append = False

        with col6:
            # Send to RAG
             if len(st.session_state.order_details) > 0 and st.button("No"):
                st.session_state.current_step = 10



    if st.session_state.current_step == 10:# RAG step
        st.write("Would you also be interested in this suggestion?")
        # prepare the input to be a proper string with the correct information that has been saved in the global variables
        rag_query = f"My Order is a {st.session_state.current_sandwich}, {st.session_state.size}, with {st.session_state.bread}, {st.session_state.cheese}, {st.session_state.veggies}, and {st.session_state.sauces}."
        suggestion = get_completion_rag(rag_query) # use AI to generated a response
        st.write(suggestion) # show the response to the user aka 'print'
        col7, col8 = st.columns(2) # They can either accept the suggestion, and add it to their order, or say no and continue to checkout
        with col7:
            if st.button("Yes Please"): # If they accept it
                dict_raw = format_sandwich(suggestion) # generate a dictionary containing the information we need (sandwich, bread, size, etc)
                proper_dict = json.loads(dict_raw) # change from str to dict
                #st.write(dict_raw) # test to see in proper dict form
                # Take the values identified by the AI to append it to the proper global values
                st.session_state.current_sandwich = proper_dict['Sandwich']
                if proper_dict['Size'] is not '':
                    st.session_state.size = proper_dict['Size']
                if proper_dict['Bread'] is not '':
                    st.session_state.bread = proper_dict['Bread']
                if proper_dict['Cheese'] is not '':
                    st.session_state.cheese = proper_dict['Cheese']
                if proper_dict['Veggies'] is not [] and proper_dict['Veggies'] is not '':
                    st.session_state.veggies = proper_dict['Veggies']
                else:
                   st.session_stae.veggies = ''
                if proper_dict['Sauces'] is not [] and proper_dict['Sauces'] is not '':
                   st.session_state.sauces = proper_dict['Sauces']
                else:
                   st.session_state.sauces = ''
                # Use the new global values to create the standard receipt format and append it too
                price = sandwiches[st.session_state.current_sandwich]["price_6"] if st.session_state.size == "6-inch" else sandwiches[st.session_state.current_sandwich]["price_footlong"]
                quantity = 1 # only recommending one sandwich
                line_total = price * quantity
                st.session_state.order_details.append({
                    "name": st.session_state.current_sandwich,
                    "size": st.session_state.size,
                    "bread": st.session_state.bread,
                    "cheese": st.session_state.cheese,
                    "veggies": st.session_state.veggies,
                    "sauces": st.session_state.sauces,
                    "quantity": quantity,
                    "line_total": line_total
                })
                st.session_state.order_complete = True # Sends us to the summary step

        with col8:
            # Complete Order
            if len(st.session_state.order_details) > 0 and st.button("No Thank You"): # if they don't want the suggestion, go straight to the summary
                st.session_state.order_complete = True


    # Complete Order
    if st.session_state.order_complete: # The summary step
        # st.write(f"Your order so far{st.session_state.order_details}") # check nothing has duplicated for admin only
        order_summary = generate_summary(username, st.session_state.order_details, st.session_state.combo) # use AI to generate the summary
        st.write(order_summary) # 'print' the summary
        st.write("Thank you for your order!")


