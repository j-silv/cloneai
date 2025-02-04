import os
import discord
from dotenv import load_dotenv
import cloneai.tts.tacotron2

load_dotenv()


# class MyClient(discord.Client):
#     async def on_ready(self):
#         print(f'Logged on as {self.user}!', flush=True)

#     async def on_voice_state_update(self, member, before, after):
#         print(f"{member.name=} changed voice state", flush=True)
        
#         if (before.channel is None):
#             text_channel = member.guild.text_channels[-1]
#             names = ["**" + member.name + "**" for member in after.channel.members]
#             grammar = "are" if len(names) > 1 else "is"
#             await text_channel.send(f"@everyone, {', '.join(names)} {grammar} in the voice chat! Join voice channel: {after.channel.jump_url}")

# intents = discord.Intents.default()
# intents.voice_states = True

# client = MyClient(intents=intents)
# client.run(TOKEN)

bot = discord.Bot()

@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")

@bot.slash_command(name="hello", description="Say hello to the bot")
async def hello(ctx: discord.ApplicationContext):
    await ctx.respond("Hey!")


@bot.slash_command(name="play")
async def play(ctx: discord.ApplicationContext, text: str):
    # Ensure the user is in a voice channel
    if ctx.author.voice is None or ctx.author.voice.channel is None:
        await ctx.respond("You need to be in a voice channel!", ephemeral=True)
        return

    vc = ctx.voice_client

    if not vc: # We firstly check if there is a voice client
        vc = await ctx.author.voice.channel.connect() # If there isn't, we connect it to the channel

    # Now we are going to check if the invoker of the command
    # is in the same voice channel than the voice client, when defined.
    # If not, we return an error message.
    if ctx.author.voice.channel.id != vc.channel.id:
        return await ctx.respond("You must be in the same voice channel as the bot.")

    outfile = "./data/test/tts.wav"

    cloneai.tts.tacotron2.run(text, outfile)

    # WAV file to play (replace with your actual file path)
    audio_source =  discord.FFmpegPCMAudio(outfile, executable="ffmpeg")

    # Ensure the bot is not already playing something
    if not vc.is_playing():
        vc.play(audio_source, after=lambda e: print("Finished playing."))
        await ctx.respond("Playing your audio!")
    else:
        await ctx.respond("Already playing audio!")


bot.run(os.getenv('DISCORD_TOKEN'))