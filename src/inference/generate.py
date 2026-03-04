import torch

from src.codec.decode import decode_tokens
from src.models.vocab import CombinedVocab


def generate_audio(model, text: str, out_wav: str, cfg: dict, speaker_id: int = 0):
    device = cfg.get('device', 'cpu')
    model.to(device)
    model.eval()
    vocab = CombinedVocab()

    prompt = vocab.encode_text(text)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)

    inf = cfg.get('inference', {})
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=inf.get('max_new_tokens', 64),
            do_sample=True,
            top_k=inf.get('top_k', 20),
            temperature=inf.get('temperature', 1.0),
            pad_token_id=vocab.text_tokenizer.eos_token_id,
        )

    generated = out[0].tolist()[len(prompt):]
    audio_tokens = vocab.decode_audio(generated)
    token_path = out_wav.replace('.wav', '.pt')
    torch.save(torch.tensor(audio_tokens, dtype=torch.long), token_path)
    decode_tokens(token_path, out_wav, bandwidth=cfg.get('codec', {}).get('bandwidth', 6.0), device=device)
    return out_wav
