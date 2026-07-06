
import tkinter as tk
from tkinter import ttk, messagebox
import csv
import os
import re
import hashlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "reference_papers.csv"

# ---------------- COLOR PALETTE ----------------
SIDEBAR_BG = "#1b1f3b"
SIDEBAR_ACTIVE = "#2e335a"
CONTENT_BG = "#f4f6fa"
CARD_BG = "#ffffff"
ACCENT = "#5c6bc0"
ACCENT_DARK = "#3f4899"
SUCCESS = "#2ecc71"
WARNING = "#f39c12"
DANGER = "#e74c3c"
TEXT_DARK = "#22273a"
TEXT_MUTED = "#707585"
TEXT_LIGHT = "#ecf0f1"
HIGHLIGHT_BG = "#ffe08a"


# AI ENGINE PART 1: Text Similarity using Scikit-learn

def compute_similarity_scores(student_text, reference_texts):
    """
    Uses scikit-learn's TfidfVectorizer + cosine_similarity to score how
    similar the student's paper is to each reference paper.
    Returns a list of similarity percentages (0-100), one per reference paper.
    """
    all_texts = reference_texts + [student_text]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    student_vector = tfidf_matrix[-1]          # last row = student paper
    reference_vectors = tfidf_matrix[:-1]      # all rows except the last

    similarities = cosine_similarity(student_vector, reference_vectors)[0]
    return [round(score * 100, 1) for score in similarities]


# AI ENGINE PART 2: Document Fingerprinting (for phrase highlighting)

def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def generate_fingerprints(text, k=6):
    """
    Document Fingerprinting: breaks text into overlapping k-word chunks
    ("k-grams"), then hashes each chunk into a short fingerprint.
    Two documents that share fingerprints share exact phrases - this is the
    same core idea used by real plagiarism engines like MOSS.
    Returns: dict mapping {hash: phrase}
    """
    words = tokenize(text)
    fingerprints = {}
    for i in range(len(words) - k + 1):
        phrase = " ".join(words[i:i + k])
        fingerprint_hash = hashlib.md5(phrase.encode()).hexdigest()
        fingerprints[fingerprint_hash] = phrase
    return fingerprints


def find_matching_phrases(student_text, reference_text, k=6):
    """Returns the set of exact phrases shared between two texts via fingerprint matching."""
    student_fp = generate_fingerprints(student_text, k)
    reference_fp = generate_fingerprints(reference_text, k)

    shared_hashes = set(student_fp.keys()) & set(reference_fp.keys())
    return {student_fp[h] for h in shared_hashes}


# DATA HANDLING

def load_papers():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_paper(title, text):
    file_exists = os.path.exists(DATA_FILE)
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Title", "Text"])
        writer.writerow([title, text])



# GUI APPLICATION

class PlagiarismApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Research Paper Plagiarism Detector")
        self.root.geometry("980x680")
        self.root.configure(bg=CONTENT_BG)

        self.setup_styles()
        self.build_layout()
        self.show_page("add")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=CARD_BG, fieldbackground=CARD_BG,
                         foreground=TEXT_DARK, rowheight=28, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background=ACCENT, foreground="white",
                         font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", ACCENT)])

    def build_layout(self):
        sidebar = tk.Frame(self.root, bg=SIDEBAR_BG, width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="🔍 PlagCheck AI", bg=SIDEBAR_BG, fg="white",
                 font=("Segoe UI", 16, "bold"), pady=25).pack(fill="x")

        self.nav_buttons = {}
        nav_items = [
            ("add", "➕  Add Reference Paper"),
            ("view", "📚  Reference Library"),
            ("check", "🕵️  Check Plagiarism"),
        ]
        for key, label in nav_items:
            btn = tk.Button(sidebar, text=label, anchor="w", bg=SIDEBAR_BG, fg=TEXT_LIGHT,
                             font=("Segoe UI", 11), bd=0, padx=20, pady=14,
                             activebackground=SIDEBAR_ACTIVE, activeforeground="white",
                             command=lambda k=key: self.show_page(k))
            btn.pack(fill="x")
            self.nav_buttons[key] = btn

        footer = tk.Label(sidebar, text="AI: Scikit-learn TF-IDF\n+ Document Fingerprinting",
                           bg=SIDEBAR_BG, fg="#7d84b8", font=("Segoe UI", 8, "italic"), justify="left")
        footer.pack(side="bottom", pady=20, padx=20, anchor="w")

        self.content = tk.Frame(self.root, bg=CONTENT_BG)
        self.content.pack(side="right", fill="both", expand=True)

        self.pages = {}
        self.build_add_page()
        self.build_view_page()
        self.build_check_page()

    def show_page(self, key):
        for k, btn in self.nav_buttons.items():
            btn.configure(bg=SIDEBAR_ACTIVE if k == key else SIDEBAR_BG)
        for k, frame in self.pages.items():
            frame.pack_forget()
        self.pages[key].pack(fill="both", expand=True, padx=30, pady=25)
        if key == "view":
            self.refresh_library_list()

    def make_card(self, parent, title, subtitle=None):
        card = tk.Frame(parent, bg=CARD_BG, padx=25, pady=25,
                         highlightbackground="#dfe3ee", highlightthickness=1)
        card.pack(fill="both", expand=True)
        tk.Label(card, text=title, bg=CARD_BG, fg=TEXT_DARK,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")
        if subtitle:
            tk.Label(card, text=subtitle, bg=CARD_BG, fg=TEXT_MUTED,
                     font=("Segoe UI", 9), wraplength=650, justify="left").pack(anchor="w", pady=(2, 15))
        else:
            tk.Frame(card, bg=CARD_BG, height=10).pack()
        return card

    # ---------- Page: Add Reference Paper ----------
    def build_add_page(self):
        page = tk.Frame(self.content, bg=CONTENT_BG)
        self.pages["add"] = page
        card = self.make_card(page, "Add a Reference Paper",
                               "Add existing papers/articles here first (standing in for the "
                               "online sources a real paper would be checked against). New "
                               "student submissions will be checked against everything you add here.")

        tk.Label(card, text="Paper Title", bg=CARD_BG, fg="#636e72", font=("Segoe UI", 9)).pack(anchor="w")
        self.title_entry = tk.Entry(card, font=("Segoe UI", 10), relief="solid", bd=1)
        self.title_entry.pack(fill="x", ipady=5, pady=(3, 15))

        tk.Label(card, text="Paper Text", bg=CARD_BG, fg="#636e72", font=("Segoe UI", 9)).pack(anchor="w")
        text_frame = tk.Frame(card)
        text_frame.pack(fill="both", expand=True, pady=(3, 15))
        self.paper_text = tk.Text(text_frame, wrap="word", font=("Segoe UI", 10),
                                   relief="solid", bd=1, height=15)
        scrollbar = tk.Scrollbar(text_frame, command=self.paper_text.yview)
        self.paper_text.configure(yscrollcommand=scrollbar.set)
        self.paper_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        button_row = tk.Frame(card, bg=CARD_BG)
        button_row.pack(anchor="w")

        tk.Button(button_row, text="💾 Save to Reference Library", bg=ACCENT, fg="white",
                  font=("Segoe UI", 11, "bold"), bd=0, padx=20, pady=10,
                  activebackground=ACCENT_DARK, cursor="hand2",
                  command=self.save_reference_paper).pack(side="left")

        tk.Button(button_row, text="🗑 Clear", bg="#e4e7f5", fg=TEXT_DARK,
                  font=("Segoe UI", 11), bd=0, padx=20, pady=10,
                  activebackground="#d0d4ea", cursor="hand2",
                  command=self.clear_add_form).pack(side="left", padx=(10, 0))

    def clear_add_form(self):
        self.title_entry.delete(0, tk.END)
        self.paper_text.delete("1.0", "end")

    def save_reference_paper(self):
        title = self.title_entry.get().strip()
        text = self.paper_text.get("1.0", "end").strip()

        if not title or not text:
            messagebox.showwarning("Missing Info", "Please provide both a title and paper text.")
            return
        if len(text.split()) < 20:
            messagebox.showwarning("Too Short", "Please paste a more complete paper (at least ~20 words).")
            return

        save_paper(title, text)
        messagebox.showinfo("Saved", f"'{title}' added to the reference library!")
        self.clear_add_form()

    # ---------- Page: View Library ----------
    def build_view_page(self):
        page = tk.Frame(self.content, bg=CONTENT_BG)
        self.pages["view"] = page
        card = self.make_card(page, "Reference Library",
                               "All papers currently stored for comparison.")

        columns = ("Title", "WordCount")
        self.lib_tree = ttk.Treeview(card, columns=columns, show="headings", height=15)
        self.lib_tree.heading("Title", text="Paper Title")
        self.lib_tree.heading("WordCount", text="Word Count")
        self.lib_tree.column("Title", width=450)
        self.lib_tree.column("WordCount", width=120, anchor="center")
        self.lib_tree.pack(fill="both", expand=True, pady=(0, 10))

        tk.Button(card, text="🔄 Refresh", bg="#e4e7f5", fg=TEXT_DARK, bd=0,
                  font=("Segoe UI", 10), padx=15, pady=6,
                  command=self.refresh_library_list).pack(anchor="e")

    def refresh_library_list(self):
        for row in self.lib_tree.get_children():
            self.lib_tree.delete(row)
        for paper in load_papers():
            word_count = len(paper["Text"].split())
            self.lib_tree.insert("", "end", values=(paper["Title"], word_count))

    # ---------- Page: Check Plagiarism ----------
    def build_check_page(self):
        page = tk.Frame(self.content, bg=CONTENT_BG)
        self.pages["check"] = page
        card = self.make_card(page, "Check a Student Paper",
                               "Paste the new paper below. It will be compared against every paper "
                               "in your Reference Library using Scikit-learn's TF-IDF + Cosine "
                               "Similarity, and matching phrases will be found using Document "
                               "Fingerprinting.")

        text_frame = tk.Frame(card)
        text_frame.pack(fill="both", expand=True, pady=(0, 15))
        self.check_text = tk.Text(text_frame, wrap="word", font=("Segoe UI", 10),
                                   relief="solid", bd=1, height=10)
        scrollbar = tk.Scrollbar(text_frame, command=self.check_text.yview)
        self.check_text.configure(yscrollcommand=scrollbar.set)
        self.check_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.check_text.tag_configure("match", background=HIGHLIGHT_BG)

        check_button_row = tk.Frame(card, bg=CARD_BG)
        check_button_row.pack(anchor="w")

        tk.Button(check_button_row, text="🕵️ Check for Plagiarism", bg=ACCENT, fg="white",
                  font=("Segoe UI", 11, "bold"), bd=0, padx=20, pady=10,
                  activebackground=ACCENT_DARK, cursor="hand2",
                  command=self.run_plagiarism_check).pack(side="left")

        tk.Button(check_button_row, text="🗑 Clear", bg="#e4e7f5", fg=TEXT_DARK,
                  font=("Segoe UI", 11), bd=0, padx=20, pady=10,
                  activebackground="#d0d4ea", cursor="hand2",
                  command=self.clear_check_form).pack(side="left", padx=(10, 0))

        self.result_label = tk.Label(card, text="", bg=CARD_BG, font=("Segoe UI", 13, "bold"),
                                      justify="left")
        self.result_label.pack(anchor="w", pady=(15, 5))

        self.score_tree = ttk.Treeview(card, columns=("Paper", "Similarity"), show="headings", height=5)
        self.score_tree.heading("Paper", text="Reference Paper")
        self.score_tree.heading("Similarity", text="Similarity %")
        self.score_tree.column("Paper", width=450)
        self.score_tree.column("Similarity", width=120, anchor="center")

    def clear_check_form(self):
        self.check_text.tag_remove("match", "1.0", "end")
        self.check_text.delete("1.0", "end")
        self.result_label.config(text="")
        self.score_tree.pack_forget()

    def run_plagiarism_check(self):
        student_text = self.check_text.get("1.0", "end").strip()
        if not student_text or len(student_text.split()) < 10:
            messagebox.showwarning("Too Short", "Please paste a more complete paper to check (at least ~10 words).")
            return

        papers = load_papers()
        if not papers:
            messagebox.showwarning("Empty Library", "Please add at least one reference paper first.")
            return

        self.check_text.tag_remove("match", "1.0", "end")

        reference_texts = [p["Text"] for p in papers]
        titles = [p["Title"] for p in papers]

        # --- AI Step 1: Scikit-learn TF-IDF + Cosine Similarity ---
        scores = compute_similarity_scores(student_text, reference_texts)

        results = list(zip(titles, scores, reference_texts))
        results.sort(key=lambda r: r[1], reverse=True)

        self.score_tree.pack(fill="both", expand=True, pady=(5, 0))
        for row in self.score_tree.get_children():
            self.score_tree.delete(row)
        for title, sim, _ in results:
            self.score_tree.insert("", "end", values=(title, f"{sim:.1f}%"))

        # --- AI Step 2: Document Fingerprinting for phrase highlighting ---
        top_title, top_score, top_text = results[0]
        matching_phrases = find_matching_phrases(student_text, top_text, k=6)
        self._highlight_matches(student_text, matching_phrases)

        if top_score >= 60:
            color, verdict = DANGER, "🔴 High Plagiarism Risk"
        elif top_score >= 30:
            color, verdict = WARNING, "🟠 Moderate Similarity - Review Recommended"
        else:
            color, verdict = SUCCESS, "🟢 Low Similarity - Looks Original"

        self.result_label.config(
            text=f"{verdict}\nHighest match: \"{top_title}\"  —  {top_score:.1f}% similarity",
            fg=color
        )

    def _highlight_matches(self, student_text, matching_phrases):
        """Highlights words in the student text box that belong to a matched fingerprint phrase."""
        if not matching_phrases:
            return
        words_to_highlight = set()
        for phrase in matching_phrases:
            words_to_highlight.update(phrase.split())

        content = self.check_text.get("1.0", "end")
        for match in re.finditer(r"[A-Za-z0-9']+", content):
            word = match.group().lower()
            if word in words_to_highlight:
                start = f"1.0+{match.start()}c"
                end = f"1.0+{match.end()}c"
                self.check_text.tag_add("match", start, end)

# RUN APP

if __name__ == "__main__":
    root = tk.Tk()
    app = PlagiarismApp(root)
    root.mainloop()
