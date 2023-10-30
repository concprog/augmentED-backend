# AugmentED: AI Augmented Learning Library - Backend

## Description
WIP - Nothing is finished yet  

AI:
- Vector index querying almost working - for pdfs
- custom retrieval and response synthesis next
- ReAct later.. probably  

Backend: 
- User auth is done
- working on conversations
- need to make file uploading work and save in sql
- SQL schema for everything


## Setup

This project uses FastAPI, SQLAlchemy, llama_index and llama_cpp_python, and it requires Python 3.10+.

To start the application, navigate to the project folder and execute the command:

python main.py

The application will be available at `http://localhost:5555`.

## Key Features(TBD)

- User registration and authentication system with OAuth2.
- Secure conversation endpoint that generates a unique, AI-generated response based on the  question.
- Uploading books to the library (stored in SQL)
- Making custom notes for topics using AI
- Same for PYQa as well
- SQLite database for user data persistence.

## Environment Variables

To run this project, you will need to do the following:  
  
  1. Clone the project repo or download zip and extract it   
   `git clone https://github.com/nusaturn/augmentED-backend`
  2. Create a venv: `python -m venv ml-env`
  3. Install requirements: `pip install -r requirements.txt`

