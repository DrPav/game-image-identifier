import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://dl.dropboxusercontent.com/s/kls6k7nnb1bslmm/resnet-50-stage-2-2020-04-13.pkl'
export_file_name = 'resnet-50-stage-2-2020-04-13.pkl'

classes = ['anthem',
  'apex_legends',
  'borderlands_3',
  'call_of_duty_modern_warfare_2019',
  'civilization_6',
  'days_gone',
  'destiny_2',
  'dota_2',
  'fifa_20',
  'fortnite',
  'grand_theft_auto_v',
  'hearthstone',
  'kingdom_hearts_iii',
  'league_of_legends',
  'luigi’s_mansion_3',
  'madden_nfl_20',
  'mario_kart_8',
  'minecraft',
  'monster_hunter_world',
  'mortal_kombat_11',
  'nba_2k20',
  'new_super_mario_bros_u_deluxe',
  'playerunknown’s_battlegrounds',
  'rainbow_six_siege',
  'red_dead_redemption_ii',
  'resident_evil_2_2019',
  'sekiro_shadows_die_twice',
  'star_wars_jedi_fallen_order',
  'super_smash_bros_ultimate',
  'the_elder_scrolls_online',
  'the_outer_worlds',
  'tom_clancy_the_division_2',
  'total_war_three_kingdoms',
  'untitled_goose_game',
  'warframe']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
