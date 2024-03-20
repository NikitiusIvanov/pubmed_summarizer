# 🤖 LLM PubMed Summarizer Telegram bot

🧑‍💻 Telegram bot connected with Google AI API (Gemini) for summarizing free (may require subscription for some) full-text PubMed articles.

🌐 Live demo: https://t.me/pubmed_summary_bot (It's on a free hosting platform, so it might be laggy.)

## 🧠 Main Logic:
* Start chat with user.
* Request query and when it's entered, perform search in PubMed for free full-text articles with meta-analyses and systematic reviews.
* Perform search and return results as titles of the found articles in a poll format.
* Take the user's vote results with selected articles and start grabbing texts from the articles.
* Send results to Gemini-LLM for summarization with a prepared prompt.
* Take summaries from Gemini, format them, and return them to the chat.


## 🫀 Stack:
* asyncio - Telegram bot API async wrapper
* BeautifulSoup - For parsing HTML with search results and articles from pubmed.ncbi.nlm.nih.gov
* google.generativeai - Google API for summarizing with Google's LLM model Gemini
* PythonAnywhere - Deployment
