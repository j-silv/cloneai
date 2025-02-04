# bot.py
import os

import discord
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')


class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!', flush=True)

    async def on_voice_state_update(self, member, before, after):
        print(f"{member.name=} changed voice state", flush=True)
        
        if (before.channel is None):
            text_channel = member.guild.text_channels[-1]
            names = ["**" + member.name + "**" for member in after.channel.members]
            grammar = "are" if len(names) > 1 else "is"
            await text_channel.send(f"@everyone, {', '.join(names)} {grammar} in the voice chat! Join voice channel: {after.channel.jump_url}")

intents = discord.Intents.default()
intents.voice_states = True

client = MyClient(intents=intents)
client.run(TOKEN)
