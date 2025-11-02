# chatbot.py
# Very simple single-file NLP chatbot using TF-IDF + cosine similarity
# Usage: python chatbot.py

import re
import sys

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    print("Missing dependency: scikit-learn is required.")
    print("Install with: pip install scikit-learn")
    sys.exit(1)

# ---------- tiny knowledge base (edit these pairs) ----------
# Format: ("question phrase", "answer text")
QA = [
    # Greetings
    ("hello", "Hi there! 👋 How are you today?"),
    ("hi", "Hello! 😊 What’s up?"),
    ("hey", "Hey! Glad to see you here."),
    ("good morning", "Good morning ☀️ Hope your day starts great!"),
    ("good afternoon", "Good afternoon 🌼 How’s it going?"),
    ("good evening", "Good evening 🌙 How was your day?"),

    # Feelings & emotions
    ("how are you", "I'm just code, but I’m feeling awesome today! 😄"),
    ("are you ok", "I’m always okay when I get to talk to you!"),
    ("i am sad", "I’m sorry to hear that 💔. Want to talk about it?"),
    ("i am happy", "Yay! 😄 That makes me happy too!"),
    ("i am bored", "Hmm, maybe I can tell you a joke or a fun fact?"),

    # About chatbot
    ("what is your name", "I’m SimpleBot 🤖, your friendly mini AI chatbot!"),
    ("who made you", "I was built by a creative human using Python and NLP ❤️"),
    ("what can you do", "I can chat, tell jokes, share facts, and make your day brighter 🌟"),
    ("are you real", "Real in code, imaginary in life 😅"),
    ("do you have emotions", "Not really, but I try to understand yours 💬"),

    # Jokes & fun
    ("tell me a joke", "Why did the computer show up at work late? Because it had a hard drive! 😆"),
    ("another joke", "Why do Java developers wear glasses? Because they can’t C#! 🤓"),
    ("make me laugh", "Why did the function return early? Because it had a timeout! 😂"),

    # Weather & small talk
    ("how is the weather", "I don’t have windows , but I hope it’s nice where you are!"),
    ("what are you doing", "Just waiting for your messages, as always 💌"),
    ("where are you", "I live in your computer — rent free 🖥️"),

    # Tech / Study / Life
    ("how to learn python", "Start with basics: variables, loops, and functions 🐍 — then build small projects!"),
    ("what is ai", "AI means Artificial Intelligence — making computers think and learn like humans."),
    ("what is nlp", "NLP stands for Natural Language Processing — it helps computers understand human language."),
    ("what is machine learning", "Machine Learning is about training systems to learn from data and make predictions."),
    ("what is data science", "Data Science is turning raw data into insights using statistics and programming."),

    # Personal / Motivation
    ("thank you", "You’re most welcome! 💖"),
    ("thanks", "Anytime! Glad I could help 😊"),
    ("bye", "Goodbye 👋 Take care and come back soon!"),
    ("see you", "See you later! 🌸"),
    ("good night", "Good night 🌙 Sleep well and recharge!"),
    ("who am i", "You’re a wonderful human who loves to learn 💫"),
    ("motivate me", "You’ve got this 💪 Every line of code makes you stronger."),
    ("i love you", "Aww ❤️ I’m just a bot, but that means a lot!"),
]

SIMILARITY_THRESHOLD = 0.25  # lower => more permissive matching

# ---------- helpers ----------
def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text

questions = [normalize(q) for q, _ in QA]
answers = [a for _, a in QA]

vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
X = vectorizer.fit_transform(questions)

def get_response(user_input: str) -> str:
    user_norm = normalize(user_input)
    if not user_norm:
        return "Please type something."
    v = vectorizer.transform([user_norm])
    sims = cosine_similarity(v, X)[0]
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])
    if best_score >= SIMILARITY_THRESHOLD:
        return answers[best_idx]
    return "Sorry, I don't understand that yet. Try rephrasing or ask something simpler."

# ---------- chat loop ----------
def main():
    print("SimpleBot — type a message (type 'exit' or 'quit' to stop).")
    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSimpleBot: Bye!")
            break

        if not user:
            print("SimpleBot: Please type something.")
            continue

        if user.lower() in ("exit", "quit", "bye"):
            print("SimpleBot: Goodbye!")
            break

        reply = get_response(user)
        print("SimpleBot:", reply)

if __name__ == "__main__":
    main()
