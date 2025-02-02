# Twitter Plugin for GAME SDK

The Twitter plugin is a lightweight wrapper over commonly-used twitter API calls. It can be used as a executable on its own or by combining multiple of these into an executable.

## Installation
From this directory (`twitter`), run the installation:
```bash
pip install -e .
```

## Usage
1. If you don't already have one, create a X (twitter) account and navigate to the [developer portal](https://developer.x.com/en/portal/dashboard).
2. Create a project app, generate the following credentials and set the following environment variables (e.g. using a `.bashrc` or a `.zshrc` file):
  - `TWITTER_API_KEY`
  - `TWITTER_API_SECRET_KEY`
  - `TWITTER_ACCESS_TOKEN`
  - `TWITTER_ACCESS_TOKEN_SECRET`

3. Import and initialize the plugin to use in your worker:
```python
import os
from twitter_plugin_gamesdk.twitter_plugin import TwitterPlugin

# Define your options with the necessary credentials
options = {
    "id": "test_twitter_worker",
    "name": "Test Twitter Worker",
    "description": "An example Twitter Plugin for testing.",
    "credentials": {
        "apiKey": os.environ.get("TWITTER_API_KEY"),
        "apiSecretKey": os.environ.get("TWITTER_API_SECRET_KEY"),
        "accessToken": os.environ.get("TWITTER_ACCESS_TOKEN"),
        "accessTokenSecret": os.environ.get("TWITTER_ACCESS_TOKEN_SECRET"),
    },
}
# Initialize the TwitterPlugin with your options
twitter_plugin = TwitterPlugin(options)

# Post a tweet
post_tweet_fn = twitter_plugin.get_function('post_tweet')
post_tweet_fn("Hello world!")
```
You can refer to `test_twitter.py` for more examples on how to call the twitter functions.