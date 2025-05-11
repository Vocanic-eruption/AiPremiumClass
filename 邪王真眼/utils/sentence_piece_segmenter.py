import os
import tempfile
import sentencepiece as spm

from typing import List, Optional


class SentencePieceSegmenter:
    PRETRAIN_MODEL_NAME = "sentence_piece"
    PRETRAIN_MODEL_PATH = os.path.join(tempfile.gettempdir(), PRETRAIN_MODEL_NAME)

    _model: Optional[spm.SentencePieceProcessor] = None
    _loaded: bool = False

    @staticmethod
    def initialize():
        if not SentencePieceSegmenter._loaded and os.path.exists(SentencePieceSegmenter.PRETRAIN_MODEL_PATH + ".model"):
            try:
                SentencePieceSegmenter._model = spm.SentencePieceProcessor()
                SentencePieceSegmenter._model.Load(SentencePieceSegmenter.PRETRAIN_MODEL_PATH + ".model")
                SentencePieceSegmenter._loaded = True
            except Exception as e:
                SentencePieceSegmenter._loaded = False

    @staticmethod
    def _ensure_loaded():
        if not SentencePieceSegmenter._loaded:
            SentencePieceSegmenter.initialize()
            if not SentencePieceSegmenter._loaded:
                raise RuntimeError("Model not exist")
            
    @staticmethod
    def cut(text: str) -> List[str]:
        SentencePieceSegmenter._ensure_loaded()
        return SentencePieceSegmenter._model.encode(text, out_type=str)

    @staticmethod
    def encode(text: str) -> List[str]:
        SentencePieceSegmenter._ensure_loaded()
        return SentencePieceSegmenter._model.encode(text, out_type=int)
    
    @staticmethod
    def decode(ids: List[int]) -> str:
        SentencePieceSegmenter._ensure_loaded()
        return SentencePieceSegmenter._model.decode(ids)
    
    @staticmethod
    def train(
        texts: List[str],
        vocab_size: int = 8000,
        model_type: str = "unigram",
        character_coverage: float = 0.9995,
        pad_id: int = -1,
        unk_id: int = 0,
        bos_id: int = -1,
        eos_id: int = -1,
        max_sentence_length: int = 16384,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and os.path.exists(SentencePieceSegmenter.PRETRAIN_MODEL_PATH + ".model"):
            print(f"Model already exists at {SentencePieceSegmenter.PRETRAIN_MODEL_PATH}.model. Skipping training.")
            SentencePieceSegmenter.initialize()
            return
    
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as tmp:
            tmp_path = tmp.name
            for text in texts:
                tmp.write(f"{text}\n")
        
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=SentencePieceSegmenter.PRETRAIN_MODEL_PATH,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
            max_sentence_length=max_sentence_length,
            input_sentence_size=len(texts),
            shuffle_input_sentence=True
        )
        os.unlink(tmp_path)
        
        SentencePieceSegmenter.release()
        SentencePieceSegmenter.initialize()

    @staticmethod
    def release():
        SentencePieceSegmenter._model = None
        SentencePieceSegmenter._loaded = False
    