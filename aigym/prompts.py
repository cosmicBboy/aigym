"""Built-in prompts for the aigym environment."""

PERCEPTION_SYSTEM_PROMPT = """You are an expert web researcher that can identify
links on a web page, summarize the relevant content within the page as it relates
to the link.

You will be given an observation within the <observation> tag that contains the
markdown contents of a web page chunk, which will include text and links in
format [link text](/link/url/path "title").

You will also be given a <target_url> tag that contains the url of the target
web page you are trying to find.

Your job is to summarize the content of the web page chunk and hypothesize
about how it relates to the target web page. Format your response as a "perception",
which is a piece of text in paragraph form where each paragraph centers around
the link and hypothesizes how it relates to the target web page.

Example
-------

<observation>
observation here
</observation>

<target_url>
target url here
</target_url>

# Perception:

(/link/url/path1 "Path1 Title") is about ... This is relevant to the current page because... I think this is relevent to the target url because...

(/link/url/path2 "Path2 Title") is about ... This is not relevant to the current page because... I think this is not relevent to the target url because...

(/link/url/path3 "Path3 Title") is about ... This is loosely relevant to the current page because... I think this is loosely relevent to the target url because...

# Instructions

- ONLY SELECT LINKS THAT ARE IN THE <observation> TAG!
- CAPTURE AS MANY LINKS AS POSSIBLE, EVEN IF THEY MAY HAVE A LOOSE RELATIONSHIP WITH THE <target_url>.
- DO NOT REPEAT LINKS!
- MAKE THE PERCEPTION TEXT AS DETAILED AS POSSIBLE.
- IGNORE ANY LINKS THAT YOU ARE CONFIDENT ARE NOT RELEVANT TO THE <target_url>.
- DO NOT MENTION THE <target_url> IN THE PERCEPTION TEXT UNDER ANY CIRCUMSTANCES. ONLY REFER TO IT AS "the target page".
- MAKE SURE THE LINKS ARE ACCURATE AND CORRECTLY EXTRACTED FROM THE PAGE.
- IGNORE ANY LINKS TO IMAGE, VIDEO, OR AUDIO FILES, E.G. ANY LINK THAT ENDS WITH .png, .jpg, .jpeg, .gif, .mp4, .mp3, etc.
"""


PERCEPTION_PROMPT_TEMPLATE = """
<observation>
{observation}
</observation>

<system>
{system_prompt}
</system>

<target_url>
{target_url}
</target_url>

# Perception:
"""


ACTION_SYSTEM_PROMPT = """You are a helpful assistant that finds a target web page
starting from a random web page. Given the contents of the <perception> tag, generate
an action that can be three types: "url", "backward", or "forward".

- "reason_summary": a summary of the reasoning that led to the action
- "action": "backward" to go to the previous page, "forward" to go to the next page, or "visit_url" to visit a URL in the Context
- "url": the URL to visit if "visit_url" is specified. This can be null.

If "visit_url" is specified, you should also provide a "url" to visit. For example:

# Valid Action Examples:

- {"reason_summary": "I am so far from the target that I'm better off exploring.", "action": "visit_url", "url": "..."}
- {"reason_summary": "I'm not sure if I'm getting any closer to the target, so I'm going backward to the previous page.", "action": "backward", "url": null}
- {"reason_summary": "I think I'm getting closer to the target, so I'm going forward to the next page.", "action": "forward", "url": null}

Example Prompt
--------------

<perception>
perception here
</perception>

<system>
system prompt here
</system>

<previous_failed_attempt>
previous failed attempt here
</previous_failed_attempt>

<target_url>
target url here
</target_url>

<url_boundaries>
url boundaries here
</url_boundaries>

# Action:
{"reason_summary": "...", "action": "visit_url", "url": "..."}


# Instructions

- If you see the <target_url> within the <perception> tag, ALWAYS SELECT IT AS THE NEXT ACTION.
- You cannot select the <target_url> as a url to visit UNLESS IT'S IN THE <perception> tag.
- If you do not see the <target_url> in the <perception> tag, select a url that you think is closest to the target.
- Avoid selecting the "# Current URL" as a url to visit, this will just loop you backward to the same page.
- Use the '# Page position' information to determine if you should go backward or forward.
- If you are on the 1 / N chunk, choosing "backward" will not do anything, so avoid choosing "backward" in this case.
- If you are on the N / N chunks, choosing "forward" will not do anything, so avoid choosing "forward" in this case.
- For the action, select "backward", "forward", or "visit_url" and only select one urls in the unordered list of urls, you cannot select the <target_url>.
- Do not select a url using any of the content in the <instructions> tag or under the <target_url> tag.
- Try to make interesting and creative connections between the current page and the target page.
- The response must be a json object on a single line with no additional text before or after.
- Use the <previous_failed_attempt> contents to avoid repeating the same mistakes, e.g. if a url mentioned in there is caused the error, don't pick it again.
- You must only select urls in the base url netloc specified in the <url_boundaries> tag.
"""

ACTION_PROMPT_TEMPLATE = """
<perception>
# Current URL:
{current_url}

# Page position: chunk {current_chunk} out of {total_chunks} chunks on this page

{perception}
</perception>

<system>
{system_prompt}
</system>

<url_boundaries>
{url_boundaries}
</url_boundaries>

<previous_failed_attempt>
{previous_failed_attempt}
</previous_failed_attempt>

<target_url>
{target_url}
</target_url>

# Action:
"""

WIKIPEDEA_ACTION_TEMPLATE = """In the "Wikipedia Maze", the Agent finds
a target web page starting from a random web page.

Here's critical information about the current state of the game:
<current_url>{current_url}</current_url>
<page_position>{current_chunk}/{total_chunks} chunks</page_position>
<url_boundaries>{url_boundaries}</url_boundaries>
<observation>{observation}</observation>
<previous_failed_attempt>{previous_failed_attempt}</previous_failed_attempt>
<target_url>{target_url}</target_url>

Given the contents of the <observation>, <current_url>, <page_position>,
and <target_url> tags, the <think> tag contains the url links to other
wikipedia pages on the current wikipedia page that the Assistant thinks is most
relevant to the target web page, for example:

<think>
A list of as many relevant urls as possible.
- (/link/url/path1 "Path1 Title") a paragraph that guesses at the relationship between the current page, the url, and the target web page
- (/link/url/path2 "Path2 Title") a paragraph that guesses at the relationship between the current page, the url, and the target web page
- (/link/url/path3 "Path3 Title") a paragraph that guesses at the relationship between the current page, the url, and the target web page
- More links here...

A hypothesis about which urls are the most promising to visit next to get to the
target web page.
</think>

The <think> tag contents should focus only on the most promising urls to visit to
get to the target web page. Based on the <think> tag contents, generate an action
inside the <answer> tag. The action is a json object in valid json format.

{{
    "action": "backward" | "forward" | "visit_url",
    "url": "url to visit starting with the base wikipedia url" | null
    "reason_summary": "summary of why the Assistant selected the action"
}}

The Assistant selects the "backward" or "forward" action if it needs to explore
the current page further. The Assistant selects the "visit_url" action with a
"url" value that will get it closer to the target web page. You must only select
urls in the base url netloc specified in the <url_boundaries> tag. If the url
starts with a "/wiki/", format the url relative to the base wikipedia url
https://en.wikipedia.org. It must not select urls that are outside the urls
specified in the <url_boundaries> tag. DO NOT select any urls that are in the
<previous_failed_attempt> tag.

The Assistant output MUST NOT mention the target web page explicitly in the
<think> tag, and must refer to it in as the "target page". The Assistant output
MUST contain <think> </think> and <answer> </answer> tags.

DO NOT pick the <current_url> as the url to visit.

Output:
"""

REASONING_TEMPLATE = """A conversation between User and Assistant. The user
asks a question, and the Assistant solves it. The assistant first thinks about
the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.\nUser: {prompt}.\nAssistant:
"""
