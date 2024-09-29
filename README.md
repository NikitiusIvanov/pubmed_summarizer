# 🤖 LLM-powered Telegram bot for parsing and summarizes PubMed meta-analysis and systematic reviews

## 🧑‍💻 Telegram bot API connected with Google AI API (Gemini 1.5 Pro) for search and summarizing free full-text PubMed articles.

## 🌐 Live demo: https://t.me/pubmed_summary_bot (First message trigger deploying the service so first responce might requires a few seconds)

## 🧠 Main Logic:
* Start chat with user
* Request query and when it's entered, perform search in PubMed for free full-text articles with meta-analyses and systematic reviews
* Parsing PubMed search results and return in a telegram poll format
* Take the user's vote results with selected articles and start grabbing texts from the articles
* Send results to Gemini-LLM for summarization with a prepared prompt
* Take summaries from Gemini, format them, and return them to the chat


## 🫀 Stack:
* asyncio - Telegram bot API async wrapper
* BeautifulSoup - For parsing HTML with search results and articles from pubmed.ncbi.nlm.nih.gov
* google.generativeai - Google API for summarizing with Google's LLM model Gemini
* Google Cloud Run - Serverless deployment
