import re
import sys
import os
import time
import json
import asyncio
import logging
import requests
import google.generativeai as genai
from bs4 import BeautifulSoup
import httpx

from aiogram.webhook.aiohttp_server import (
    BaseRequestHandler,
    SimpleRequestHandler, 
    setup_application
)
from aiogram.client.default import DefaultBotProperties
from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Filter
from aiogram import F
from aiogram.utils.markdown import hbold
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    Update,
    CallbackQuery,
    PollAnswer,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from aiohttp import web
from aiogram.client.session.aiohttp import AiohttpSession


####################### get the enviroment variables ####################### 
# get the credentials from env vars
BOT_TOKEN = os.getenv('BOT_TOKEN')
GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')

BASE_WEBHOOK_URL = os.getenv('BASE_WEBHOOK_URL')
# Webserver settings
WEB_SERVER_HOST = "0.0.0.0"
# Port for incoming request
WEB_SERVER_PORT = 8080
# Path to webhook route, on which Telegram will send requests
WEBHOOK_PATH = "/webhook"


####################### google AI API settings ####################### 
genai.configure(api_key=GOOGLE_AI_API_KEY)

model_answering = genai.GenerativeModel('gemini-1.5-pro-002')

generation_config = {'temperature': 0}

PROMPT = """
You are a helpful and informative bot that help researchers to find and
summarize results from medicine and biology article's abstract that included below.
For given abstract explore all information and 
return summary text with listing all significant results 
please prefer results that represented by numbers, for example:
    "N participants in K studies were included in analisys and 
    shows factor X decrease risk of state Y for 30% (CI 25-35, p < 0.01)"
Please dont write any additional titles, oly list of results separated by "-"
ABSTRACT: {abstract}
"""

####################### PARSERS AND DATA PROCESSING #######################
PARSERS_CHAPTER = None

async def get_html_pm_search_results(
    query: str,
    n_results: int = 10,
    custom_url: str = None,
) -> str:
    """
    Takes query or url of the pubmed search and returns
    By default we get the results from this url:
        https://pubmed.ncbi.nlm.nih.gov/?term={query}&filter=simsearch2.ffrft&filter=pubt.meta-analysis&filter=pubt.systematicreview&size={n_results}
    
    That give us:
    ?term={query} -- here your query
    filter=simsearch2.ffrft -- filter only full free text
    filter=pubt.systematicreview -- filter systematicreview
    n_results -- here desired amount of results must one of: 10, 20, 50, 100, 200
    
    If you want you can use your ouwn result link with your desired filters
    for that copy it from adress bar your brauser and then 
    set in custom_url parameter (must start with https://pubmed.ncbi.nlm.nih.gov/?term=').
    """
    if int(n_results) not in [10, 20, 50, 100, 200]:
        raise ValueError('n_results must one of: 10, 20, 50, 100, 200')
    
    
    if custom_url is None:
        query = query.replace(' ', '+')
        url = (
        """
        https://pubmed.ncbi.nlm.nih.gov/
        ?term={query}
        &filter=simsearch2.ffrft
        &filter=pubt.meta-analysis
        &filter=pubt.systematicreview
        &size={n_results}
        """
        .format_map({'query': query, 'n_results': n_results})
        .replace('\n', '')
        .replace('    ', '')
    )
    
    if custom_url is not None:
        # replace n results to n_results
        pattern = r"&size=(\d+)"

        # Replace the matched digits with the given value
        custom_url = re.sub(pattern, f"&size={n_results}", custom_url)
        if custom_url.startswith('https://pubmed.ncbi.nlm.nih.gov/?term='):
            url = custom_url
        else:
            raise Exception('url must start with https://pubmed.ncbi.nlm.nih.gov/?term=')
    
    # response = requests.get(url)
    # Make an async GET request using httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Access the HTML content of the webpage
        return response.text
    else:
        print("Failed to retrieve HTML content. Status code:", response.status_code)


async def get_search_results_texts(
    query: str,
    n_results: int = 10,
    custom_link: str = None,
) -> tuple[str, list[str], list[str]]:
    
    if query is not None:
        html_content = await get_html_pm_search_results(
            query=query,
            n_results=n_results,
        )
    if custom_link is not None:
        html_content = await get_html_pm_search_results(
            query=None,
            n_results=n_results,
            custom_url=custom_link
        )
    
    # Parse the HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract desired information
    pm_ids = [x.get('href').strip('/') for x in soup.find_all('a', class_='docsum-title')]
    titles = [x.text.strip() for x in soup.find_all("a", class_="docsum-title")]
    authors = [x.text.strip() for x in soup.find_all("span", class_="docsum-authors full-authors")]
    journal_citations = [x.text.strip() for x in soup.find_all("span", class_="docsum-journal-citation full-journal-citation")]
    snippets = [x.text.strip() for x in soup.find_all("div", class_="full-view-snippet")]

    # create short version of the titles for dispaying as poll options
    dates = [
        x.split('.')[1].strip(' ')[:8]
        for x in
        journal_citations
    ]

    jornals = [
        x.split('.')[0].strip(' ')
        for x in
        journal_citations
    ]

    jornals_cut = [
        x.split('.')[0].strip(' ')[:19]
        for x in
        journal_citations
    ]

    journals_dates_cut = [
        x + '. ' + y
        for x, y in zip(jornals_cut, dates)
    ]

    journals_dates = [
        x + '. ' + y
        for x, y in zip(jornals, dates)
    ]

    titles_cut = [
        x[:68] + '; \n' + y
        for x, y in zip(
            titles,
            journals_dates_cut
        )
    ]

    titles = [
        x[:68] + '; \n' + y
        for x, y in zip(
            titles,
            journals_dates
        )
    ]

    return pm_ids, titles, titles_cut


async def get_pm_abstract_from_pm_id(
    pm_id: str
) -> str:
    pm_url = f'https://pubmed.ncbi.nlm.nih.gov/{pm_id}/'
    # response = requests.get(pm_url)
    
    # Make an async GET request using httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(pm_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Access the HTML content of the webpage
        html_content = response.text
    else:
        print(pm_url)
        print("Failed to retrieve HTML content. Status code:", response.status_code)

    # Parse the HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Find the abstract content
    abstract_content = soup.find('div', class_='abstract-content selected')

    abstract_text = [x.text.strip() for x in abstract_content.find_all('p')]

    # Combine all text into a single string
    abstract_text = '\n'.join(abstract_text)

    abstract_text = '\n'.join(
        [
            x for x in abstract_text.split('\n') 
            if x.isspace() == False
        ]
    )

    return abstract_text


def pubmed_parsed_abstracts_texts_joining(
    pm_ids: list[str],
    titles: list[str],
    journal_citations: list[str],
    abstracts: list[str], 
) -> str:
    """
    Takes lists of the values and returns joned string
    """
    return '\n\n'.join(
        [
            """
            PUBMED_ID: {pm_id}
            TITLE: {title}
            CITATION: {citations}
            ABSTRACT: {abstract}
            """
            .format_map(
                {
                    'pm_id': pm_id,
                    'title': title,
                    'citations': citations,
                    'abstract': abstract
                }
            )
            for pm_id,
                title,
                citations,
                abstract
            in zip(
                pm_ids,
                titles,
                journal_citations,
                abstracts
            )
        ]
    )
 

async def get_summary_from_abstract(
    abstract: str,
    model_answering: genai.GenerativeModel,
    generation_config: dict,
    prompt_to_summarize_pubmed_abstracts: str = PROMPT,
    n_tryes: int = 3,
) -> str:

    try:
        summary = await model_answering.generate_content_async(
            contents=prompt_to_summarize_pubmed_abstracts.format_map({'abstract': abstract}),
            generation_config=generation_config
        )

        result = ' '.join([part.text for part in summary.parts])
    except:
        summary = None
        n_try = 0
        while summary is None and n_try < n_tryes:
            try:
                summary = await model_answering.generate_content_async(
                    contents=prompt_to_summarize_pubmed_abstracts.format_map({'abstract': abstract}),
                    generation_config=generation_config
                )
            except:
                time.sleep(3)
                n_try += 1
    
    if summary is None: 
        result = ''
    else:
        result = ' '.join([part.text for part in summary.parts])
    if len(result) == 0:
        result = abstract

    return result


def replace_html_tags(
    text: str,
    pairs: dict[str, str] = {
        '<': '&lt;', 
        '>': '&gt;',
        '&': '&amp;'
    },
) -> str: 
    for key, value in pairs.items():
        text = text.replace(key, value)

    return text


def result_formatting(
    titles: list[str],
    pm_ids: list[str],
    summaries: list[str],
    separator: str = '-' * 50,
    max_lenght: int = 4096,
    chars_to_escape: list[str] = [
        '_', '*', '`'
    ],
    escape_symbol: str='\\'
) -> list[str]:
    """Formatting text and split into chunks"""
    # Create a regular expression pattern to match any character in the list
    pattern = r"[" + re.escape("".join(chars_to_escape)) + "]"

    # Replace matched characters with the escape symbol and the character itself
    titles = [
        re.sub(pattern, lambda match: escape_symbol + match.group(0), title)
        for title in titles
    ]
    summaries = [
        re.sub(pattern, lambda match: escape_symbol + match.group(0), summary)
        for summary in summaries
    ]

    result =  '\n'.join([
        '*' + titles[index] + '*\n' 
        +
        '_SUMMARY_: \n' 
        + summary + '\n' 
        +
        '_PUBMED ID_: ' 
        + f'[{pm_ids[index]}](https://pubmed.ncbi.nlm.nih.gov/{pm_ids[index]}/)' + '\n' 
        + separator
        for index, summary in zip(
            list(range(len(pm_ids))),
            summaries
        )
    ])

    result_lenght = len(result)
    
    if result_lenght > max_lenght:
        n_chunks = (
            (result_lenght // max_lenght) + 1 
            if 
            result_lenght % max_lenght != 0 
            else 
            (result_lenght // max_lenght)
        )
        n_results = len(result.split(separator))
        n_results_in_chunk = (n_results // n_chunks)
        
        if n_results % n_chunks != 0:
            n_results_in_chunk += 1
        
        chunks = [
            (separator).join(result.split(separator)[start: start + n_results_in_chunk])
            for start in list(range(0, n_results, n_results_in_chunk))
        ]
    else: chunks = [result]

    return chunks


def get_progress_bar(
    len_results: int,
    n_processed_results: int,
    progress_lenght: int = 20,
    filled_block: str = '▓',
    empty_block: str = '░'
) -> str:
    progress_lenght = 20
    
    n_results = len_results
    if progress_lenght % n_results == 0:
        step_size = progress_lenght // n_results
    else:
        step_size = (progress_lenght // n_results) + 1

    return (    
        f'{"".join([filled_block] * n_processed_results * step_size)}'
        f'{"".join([empty_block] * ((n_results - n_processed_results) * step_size))}'
    )


####################### TG BOT LOGIC #######################
TG_BOT_LOGIC_CHAPTER = None

async def send_pool(
    chat_id: int,
    question: str,
    options: list[str],
    BOT_TOKEN: str,
    allows_multiple_answers: bool = True,
    METHOD_NAME: str = "SendPoll",
) -> None:
    poll = {
        "chat_id":chat_id,
        "question": question,
        "options": options,
        "allows_multiple_answers": allows_multiple_answers,
        'is_anonymous': False
    }

    base_url = f'https://api.telegram.org/bot{BOT_TOKEN}/{METHOD_NAME}'
    headers = {"Content-Type": "application/json"}

    response = requests.post(base_url, headers=headers, json=poll)


class MyFilter(Filter):
    def __init__(self, my_text: str) -> None:
        self.my_text = my_text

    async def __call__(self, message: Message) -> bool:
        return message.text == self.my_text


class Form(StatesGroup):
    chat_id = State()
    query = State()
    query_results = State()
    custom_link = State()
    pm_ids = State()
    titles = State()
    titles_cut = State()
    journal_citations = State()
    abstracts = State()
    summary = State()

form_router = Router()

def keyboard_start():
    builder = ReplyKeyboardBuilder()
    builder.button(text='Perform search and summarize results')
    builder.button(text='Summarize by my link')
    builder.adjust(1)
    return builder.as_markup()

def keyboard_check_search():
    builder = ReplyKeyboardBuilder()
    builder.button(text='Summarize abstracts for all results')
    builder.button(text='Start new search')
    builder.adjust(1)
    return builder.as_markup()

def keyboard_error_link():
    builder = ReplyKeyboardBuilder()
    builder.button(text='Start new search')
    builder.button(text='Summarize by my link')
    builder.adjust(1)
    return builder.as_markup()

@form_router.message(CommandStart())
async def welcome(message: Message, state: FSMContext):
    user_name = message.from_user.first_name

    await message.answer(
        text=(
            f'*Hey, {user_name}!* \n'
            'I\'m a helpful AI summarizer for PubMed articles.\n'
            'I can perform PubMed searches based on your query to '
            'retrieve full-text results of the meta-analyses and '
            'systematic reviews. '
            'I can then summarize their main results using Gemini-LLM by Google AI.\n\n'
            'Or, you can send me a link to your search results,'
            ' and I can summarize the articles from it'
            ' Choose the option below:'
        ),
        reply_markup=keyboard_start(),
        parse_mode=ParseMode.MARKDOWN
    )

@form_router.message(F.text == 'Summarize by my link')
async def request_custom_link(message: Message, state: FSMContext):

    await state.set_state(Form.custom_link)
    await message.answer(
        'Paste your link below (must start with https://pubmed.ncbi.nlm.nih.gov/?term=)',
        reply_markup=ReplyKeyboardRemove()
    )


@form_router.message(Form.custom_link)
async def get_pubmed_results_from_custom_link(message: Message, state: FSMContext):
    custom_link = message.text
    if custom_link.startswith('https://pubmed.ncbi.nlm.nih.gov/?term='):
        output_text = 'Start performing pubmed search by your query, please wait'
        await message.answer(output_text, reply_markup=keyboard_check_search())  
    else:
        output_text = 'url must start with https://pubmed.ncbi.nlm.nih.gov/?term='
        await message.answer(output_text, reply_markup=keyboard_error_link())
        await state.clear()
        return  

    await state.update_data(custom_link=custom_link)
    await state.update_data(chat_id=message.chat.id)
    await state.set_state(Form.query_results)
      
    ##### search process and return results
    pm_ids, titles, titles_cut = await get_search_results_texts(
        query=None,
        custom_link=message.text
    )
    chat_id = message.chat.id

    await state.update_data(pm_ids=pm_ids)
    await state.update_data(titles=titles)
    await state.update_data(titles_cut=titles_cut)
    await send_pool(
        chat_id=chat_id,
        question='Select aritcles and press vote for summarizing',
        options=titles_cut,
        BOT_TOKEN=BOT_TOKEN,
    )


@form_router.message(
    F.text.in_({
        'Perform search and summarize results',
        'Start new search'
    })
)
async def request_the_query(message: Message, state: FSMContext):
    output_text = 'Past search query below'
    await state.set_state(Form.query)
    await message.answer(
        output_text,
        reply_markup=ReplyKeyboardRemove()
    )


@form_router.message(Form.query)
async def get_pubmed_results_from_query(message: Message, state: FSMContext):
    output_text = 'Start performing pubmed search by your query, please wait'
    await state.update_data(query=message.text)
    await state.update_data(chat_id=message.chat.id)
    await state.set_state(Form.query_results)
    await message.answer(output_text, reply_markup=keyboard_check_search())    
    
    ##### search process and return results
    pm_ids, titles, titles_cut = await get_search_results_texts(
        query=message.text
    )
    chat_id = message.chat.id

    await state.update_data(pm_ids=pm_ids)
    await state.update_data(titles=titles)
    await state.update_data(titles_cut=titles_cut)
    await send_pool(
        chat_id=chat_id,
        question='Select aritcles and press vote for summarizing',
        options=titles_cut,
        BOT_TOKEN=BOT_TOKEN,
    )


@form_router.poll_answer()
async def summarize_by_poll_answer_reaction(
    poll_answer: PollAnswer, 
    state: FSMContext
):
    user_id = poll_answer.user.id
    data = await state.get_data()
    chat_id = data['chat_id']
    pm_ids = data['pm_ids']
    titles = data['titles']
    if poll_answer is not None:
        chosen_options = [str(x) for x in poll_answer.option_ids]
    else:
        chosen_options = list(range(len(pm_ids)))
    
    bot = poll_answer.bot
    await bot.send_message(
        chat_id=chat_id,
        text=f"Your choices: {', '.join(chosen_options)}"
    )

    len_results = len(chosen_options)
    n_processed_results = 0

    progress_bar = get_progress_bar(
        len_results=len_results,
        n_processed_results=n_processed_results
    )
    
    message = await bot.send_message(
        chat_id=chat_id,
        text=(
            'Grabbing articles abstracts..\n'
            f'{progress_bar} {n_processed_results} / {len_results}'
        )
    )

    abstracts = []

    for index in poll_answer.option_ids:

        abstracts.append(await get_pm_abstract_from_pm_id(pm_id=pm_ids[index]))

        n_processed_results += 1

        progress_bar = get_progress_bar(
            len_results=len_results,
            n_processed_results=n_processed_results
        )

        await message.edit_text(
            text=(
                'Grabbing articles abstracts..\n'
                f'{progress_bar} {n_processed_results} / {len_results}'
            )
        )

    await state.update_data(abstracts=abstracts)

    len_results = len(chosen_options)
    n_processed_results = 0

    progress_bar = get_progress_bar(
        len_results=len_results,
        n_processed_results=n_processed_results
    )
    
    message = await bot.send_message(
        chat_id=chat_id,
        text=(
            'Summarizing..\n'
            f'{progress_bar} {n_processed_results} / {len_results}'
        )
    )

    summaries = []
    for abstract in abstracts:
        summaries.append(
            await get_summary_from_abstract(
                abstract=abstract,
                model_answering=model_answering,
                generation_config=generation_config,
            )
        )

        n_processed_results += 1

        progress_bar = get_progress_bar(
            len_results=len_results,
            n_processed_results=n_processed_results
        )

        await message.edit_text(
            text=(
                'Summarizing..'
                f'{progress_bar} {n_processed_results} / {len_results}'
            )
        )

    results = result_formatting(
        titles=titles,
        pm_ids=pm_ids,
        summaries=summaries,
    )

    for chunk in results:
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=chunk, 
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard_start()
            )
        except:
            print('Not sented:')
            print(chunk)


@form_router.message(F.text == 'Summarize abstracts for all results')
async def summarize_all_results(message: Message, state: FSMContext):
    data = await state.get_data()
    chat_id = data['chat_id']
    pm_ids = data['pm_ids']
    titles = data['titles']
    chosen_options = [str(x) for x in range(len(pm_ids))]
    
    await message.answer(text=f"Your choices: {', '.join(chosen_options)}")

    await message.answer(
        text='Start grabbing articles abstracts (it took about 10s)'
    )

    abstracts = [
        await get_pm_abstract_from_pm_id(pm_id=pm_ids[index])
        for index in range(len(pm_ids)) 
    ]

    await state.update_data(abstracts=abstracts)
    await message.answer(
        text='Start summarizing (it took about 20s)'
    )

    summaries = [
        await get_summary_from_abstract(
            abstract=abstract,
            model_answering=model_answering,
            generation_config=generation_config,
        )
        for abstract in abstracts
    ]

    results = result_formatting(
        titles=titles,
        pm_ids=pm_ids,
        summaries=summaries,
    )

    for chunk in results:
        try:
            await message.answer(
                chunk,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard_start()
            )
        except:
            print('Not sented:')
            print(chunk)


#################################### Bot settings ####################################
BOT_SETTINGS_CHAPTER = None
# Set the webhook for recieving updates in your url wia HTTPS POST with JSONs
async def on_startup(bot: Bot) -> None:
    
    # If you have a self-signed SSL certificate, then you will need to send a public
    # certificate to Telegram, for this case we'll use google cloud run service so
    # it not required to send sertificates
    await bot.set_webhook(
        f"{BASE_WEBHOOK_URL}{WEBHOOK_PATH}",
    )


async def health_check(request):
    return web.Response(text="Bot is alive!")


async def main() -> None:
    # Initialize Bot instance with a default parse mode which will be passed to all API calls
    session = AiohttpSession()
    bot = Bot(BOT_TOKEN, parse_mode=ParseMode.HTML, session=session)
    
    # And the run events dispatching
    dp = Dispatcher()
    dp.include_router(form_router)
    
    # Run the bot with long-polling
    asyncio.create_task(dp.start_polling(bot))

    # Set up aiohttp web server to serve health check
    app = web.Application()
    app.add_routes([web.get("/", health_check)])  # Dummy endpoint for health checks

    # Start aiohttp web server for Google Cloud Run's health check
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host="0.0.0.0", port=WEB_SERVER_PORT)
    await site.start()

    logging.info(f"Serving on http://0.0.0.0:{WEB_SERVER_PORT}")

    # Keep the bot and server running forever
    while True:
        await asyncio.sleep(3600)  # Just keep running the event loop

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())

# def main() -> None:
#     # Dispatcher is a root router
#     dp = Dispatcher()

#     dp.include_router(form_router)

#     # Register startup hook to initialize webhook
#     dp.startup.register(on_startup)

#     # Initialize Bot instance with a default parse mode 
#     # which will be passed to all API calls
#     bot = Bot(BOT_TOKEN, default=DefaultBotProperties())
    
#     # And the run events dispatching
#     # Create aiohttp.web.Application instance
#     app = web.Application()

#     # Create an instance of request handler,
#     # aiogram has few implementations for different cases of usage
#     # In this example we use SimpleRequestHandler 
#     # which is designed to handle simple cases
#     webhook_requests_handler = SimpleRequestHandler(
#         dispatcher=dp,
#         bot=bot,
#     )

#     # Register webhook handler on application
#     webhook_requests_handler.register(app, path=WEBHOOK_PATH)

#     # Mount dispatcher startup and shutdown hooks to aiohttp application
#     setup_application(app, dp, bot=bot)

#     # And finally start webserver
#     web.run_app(app, host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, stream=sys.stdout)
#     main()
