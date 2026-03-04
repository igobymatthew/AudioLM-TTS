from fastapi import FastAPI
from pydantic import BaseModel


class TTSRequest(BaseModel):
    text: str
    speaker_id: int = 0


app = FastAPI(title='AudioLM-TTS API')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/synthesize')
def synthesize(req: TTSRequest):
    # Stub endpoint; real generation is wired in serve.py process state.
    return {'message': 'accepted', 'text': req.text, 'speaker_id': req.speaker_id}
