#!/bin/bash

# Create .streamlit directory if it doesn't exist
mkdir -p ~/.streamlit/

# Create Streamlit config file for Heroku deployment
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
