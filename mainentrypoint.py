import streamlit as st
from multipage import MultiApp
from apps import main_app  # This is your main classification app (apps/main_app.py)

# Import your new pages
from pages import manual, team

app = MultiApp()

# Add all your applications here
app.add_app("https://github.com/unnatisrivastava952/peptide-classification-/blob/main/app2.py",main_app.app)      # Your peptide classification page
app.add_app("User Manual", manual.app)       # The manual page
app.add_app("Team", team.app)                # The team page

app.run()
