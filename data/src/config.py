import os
import re

# --- 1. PATH CONFIGURATION ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data_raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data_processed")

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# --- 2. REGEX PATTERNS ---

# A. LOOSE FILTER 
LOOSE_KEYWORDS = [
    r"\bnlp\b", r"\bnatural language\b", r"\btext\b", r"\blingui", 
    r"\btransformer\b", r"\battention\b", r"\blanguage model\b", 
    r"\bembedding\b", r"\bgeneration\b", r"\bparsing\b",
    r"\btranslat", r"\bsentiment\b", r"\bgpt\b", r"\bbert\b",
    r"\bcorpus\b", r"\bsemantic\b", r"\bword\b", r"\btoken", 
    r"\bnamed entity\b", r"\bner\b", r"\bpos tagging\b",
    r"\bdependency\b", r"\bsyntax\b", r"\bsyntactic\b",
    r"\bdialogue\b", r"\bconversation", r"\bchatbot\b",
    r"\bsummariz", r"\bquestion answering\b", r"\bqa\b",
    r"\binformation extraction\b", r"\brelation extraction\b"
]

# B. HARD EXCLUDE
HARD_EXCLUDE = [
    # Biology & Medicine
    r"\bmolecular\b", r"\bprotein\b", r"\bgenom", r"\bdna\b", r"\brna\b",
    r"\bpatient\b", r"\bsurgery\b", r"\bclinical\b", r"\bmedical imaging\b",
    r"\bcell\b", r"\btissue\b", r"\bpharmaco", r"\bdrug\b", r"\bcancer\b",
    r"\bantibod", r"\bvaccin", r"\bdiagnos", r"\btherapy\b",
    # Engineering & Physics
    r"\bhigh voltage\b", r"\belectric\b", r"\bcircuit\b", r"\bgrid\b",
    r"\bconcrete\b", r"\bmaterial\b", r"\bsteel\b", r"\bstructural\b",
    r"\bwireless\b", r"\bantenna\b", r"\bradio\b", r"\bsignal processing\b",
    r"\bseismic\b", r"\bgeolog", r"\bfluid\b", r"\bthermal\b",
    r"\bpower generation\b", r"\bturbine\b", r"\bengine\b",
    # Finance & Economics  
    r"\bstock price\b", r"\bmarket\b", r"\btrading\b", r"\bportfolio\b",
    r"\bfinancial\b", r"\beconomic\b", r"\bbanking\b",
    # Agriculture & Plants
    r"\bplant growth\b", r"\bcrop\b", r"\bsoil\b", r"\bblueberry\b",
    r"\bagricultural\b", r"\bfertiliz", r"\bharvest\b",
    # Other domains
    r"\bvideo\b.*\bgeneration\b", r"\bimage\b.*\bgeneration\b",
    r"\b3d model", r"\bcad\b", r"\brendering\b", r"\btext-to-image\b", # Thêm text-to-image
    r"\barchaeolog", r"\bhistorical artifact\b"
]

# C. SCORING PATTERNS

# Thuần NLP (+20 điểm, độ tin cậy tuyệt đối)
PURE_NLP_VENUES = [
    r"association for computational linguistics", r"\bacl\b",
    r"\bemnlp\b", r"\bnaacl\b", r"\beacl\b", r"\bcoling\b", 
    r"\bconll\b", r"\btacl\b", r"\bsemeval\b", r"\blrec\b", 
    r"\bijcnlp\b"
]

# AI/ML Tổng hợp (+5 điểm, cần kiểm tra thêm keyword)
GENERAL_AI_VENUES = [
    r"\bneurips\b", r"\biclr\b", r"\baaai\b", r"\bicml\b", r"\bcvpr\b"
]

STRONG_KEYWORDS = [
    r"\bnlp\b", r"\blarge language model\b", r"\bllm\b", r"\bllms\b", 
    r"\bchatgpt\b", r"\bgpt-[3-5]", r"\bllama\b", r"\bmistral\b",
    r"\bbert\b", r"\broberta\b", r"\bt5\b",
    r"\bword embedding\b", r"\bword2vec\b", r"\bglove\b", 
    r"\bnamed entity recognition\b", r"\bner\b",
    r"\bmachine translation\b", r"\bsentiment analysis\b",
    r"\bquestion answering\b", r"\btext summarization\b",
    r"\btext generation\b", r"\bprompt engineering\b",
    r"\brag\b", r"\bretrieval-augmented\b",
    r"\bchain-of-thought\b", r"\binstruction tuning\b"
]

CONTEXT_KEYWORDS = [
    r"\btransformer(?:s)?\b", r"\battention mechanism\b", r"\bself-attention\b",
    r"\bencoder\b", r"\bdecoder\b", r"\blstm\b", r"\brnn\b",
    r"\bseq2seq\b", r"\bsequence to sequence\b",
    r"\btext mining\b", r"\binformation extraction\b",
    r"\bpre-?train", r"\bfine-?tun"
]