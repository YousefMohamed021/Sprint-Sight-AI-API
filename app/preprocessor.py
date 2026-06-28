import re
import textstat

def preprocess_sprint_text(text: str) -> str:
    """
      1. Remove HTML tags
      2. Remove URLs
      3. Remove file references
      4. Remove inline code blocks
      5. Remove commit hashes
      6. Remove special Jira format references
      7. Remove control characters
      8. Remove excessive punctuation / special chars
      9. Collapse whitespace
    """
    if not isinstance(text, str) or text.strip() == "":
        return "[EMPTY]"

    # 1. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|ftp://\S+|www\.\S+", " ", text)

    # 3. Remove file references  (e.g. foo/bar.java, ../config.xml)
    text = re.sub(r"(?:[\.\./]*[\w\-]+/)+[\w\-\.]+\.\w+", " ", text)

    # 4. Remove inline code blocks  ```...``` or `...`
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]*`", " ", text)

    # 5. Remove commit hashes / SHA strings (6+ hex chars)
    text = re.sub(r"\b[0-9a-f]{6,}\b", " ", text)

    # 6. Remove special Jira references  {{...}}, [~user], [^attachment]
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)
    text = re.sub(r"\[~[^\]]*\]", " ", text)
    text = re.sub(r"\[\^[^\]]*\]", " ", text)

    # 7. Remove control characters  (\n \r \t)
    text = re.sub(r"[\n\r\t]+", " ", text)

    # 8. Remove excessive punctuation (keep alpha-numeric + . , ! ?)
    text = re.sub(r"[^\w\s.,!?\'\\-]", " ", text)

    # 9. Collapse multiple spaces
    text = re.sub(r" +", " ", text).strip()

    return text if text else "[EMPTY]"

def calculate_fog_index(text: str) -> float: 
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    
    fog_index = textstat.gunning_fog(text)
    return round(fog_index, 4)