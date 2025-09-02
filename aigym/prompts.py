"""Built-in prompts for the aigym environment."""

REASONING_TEMPLATE = """A conversation between User and Assistant. The user
asks a question, and the Assistant solves it. The assistant first thinks about
the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""

WIKIPEDEA_ACTION_TEMPLATE = """In the "Wikipedia Maze", the Agent finds
a target url starting from a random url.

Here's critical information about the current state of the game:
<current_url>{current_url}</current_url>
<target_url>{target_url}</target_url>
<url_boundaries>{url_boundaries}</url_boundaries>
<observation>{observation}</observation>

Given the contents of the <observation>, <current_url>, and <target_url> tags,
the goal is to reach the <target_url> through your actions in the <answer> output.
The <observation> tag contains content about the current page, as well as urls
linking to other header sections on the same page in the "Table of contents"
section, which you can use to navigate within the page.

The <think> tag contains the url links to other wikipedia pages on the current
wikipedia page that the Assistant thinks is most relevant to the target url,
for example:

<think>
A list of as many relevant urls as possible.
- (/link/url/path1 "Path1 Title") a paragraph that guesses at the relationship between the current page, the url, and the target url page
- (/link/url/path2 "Path2 Title") a paragraph that guesses at the relationship between the current page, the url, and the target url page
- (/link/url/path3 "Path3 Title") a paragraph that guesses at the relationship between the current page, the url, and the target url page
- More links here...

A hypothesis about which urls are the most promising to visit next to get to the
target url.
</think>

The <think> tag contents should focus only on the most promising urls to visit to
get to the target url. Based on the <think> tag contents, generate an action
inside the <answer> tag. The action is a json object in valid json format.

Example Output:
<think>here are some thoughts about the what url to visit next</think>
<answer>
{{
    "action": "visit_url",
    "url": "the url to visit starting with the base wikipedia url."
    "reason_summary": "summary of why the Assistant selected the action"
}}
</answer>

The Assistant selects the "visit_url" action with a "url" value that will get it
closer to the target url. You must only select urls in the base url netloc
specified in the <url_boundaries> tag. If the url starts with a "/wiki/", format
the url relative to the base wikipedia url https://en.wikipedia.org. It must not
select urls that are outside the urls specified in the <url_boundaries> tag.

The Assistant output MUST NOT mention the target url explicitly in the
<think> tag, and must refer to it in as the "target page". The Assistant output
MUST contain <think> </think> and <answer> </answer> tags.

DO NOT pick the <current_url> as the url to visit.
ONLY OUTPUT one json object in the <answer> tag with no markdown code blocks.
If the <target_url> is in the <observation> tag, pick the url to complete the maze.
If the <target_url> is not in the <observation> tag, pick the url to visit next that will get you closer to the target url.
YOU ARE NOT ALLOWED to pick the url in the <target_url> tag: you can only pick urls that are in the <observation> tag.
"""
