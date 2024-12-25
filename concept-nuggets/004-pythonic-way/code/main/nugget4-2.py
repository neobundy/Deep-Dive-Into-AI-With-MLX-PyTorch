import streamlit as st

# Define a context manager for a Streamlit form
class streamlit_form:
    def __enter__(self):
        # Start the form
        self.form = st.form(key='my_form')
        return self.form

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Submit button for the form
        submit_button = self.form.form_submit_button('Submit')
        if submit_button:
            st.success('Form submitted successfully!')

# Using the context manager for a cleaner form handling
with streamlit_form() as form:
    name = form.text_input('Name')
    age = form.number_input('Age', min_value=18, max_value=100)

# The form is automatically handled, focusing on user input collection