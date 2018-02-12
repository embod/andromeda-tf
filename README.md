# Andromeda Tensorforce Client
This library uses Tensorforce deep reinforcement learning models for controlling agents in the andromeda environment.

[Tensorforce](https://github.com/reinforceio/tensorforce) is a deep reinforcement learning library that uses tensorflow under the hood.

# Installing

## Dependencies

Embod.ai clients require python 3.4 or above

Firstly install the requirements that this agent controller requires.
```python
pip install -r requirements.txt
```

# Running

You will need create an agent and retrieve your api key before you can use this library.
You will need to create an embod.ai account to get these.

## Creating an agent and getting your API key

This is really easy...[**Click here to get started**](https://app.embod.ai/documentation/getting-started).

## Running

```python
python run.py -p [YOUR API KEY] -a [YOUR AGENT ID]
```

Once the agent is running you can view it's progress on the andromeda view page [here](https://app.embod.ai/andromeda/view)

You can also see the other agents that it is competing against here too!

# Support

Currently embod.ai is in Alpha, please email support@embod.ai for access 