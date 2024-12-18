from __future__ import annotations
import regex as re


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]*>", "", text)


def convert_unicode(text: str) -> str:
    """Convert decomposed Unicode characters to composed Vietnamese characters."""
    char1252 = (
        "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|"
        "ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|"
        "ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|"
        "À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|"
        "Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|"
        "Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ"
    )
    charutf8 = (
        "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|"
        "ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|"
        "À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|"
        "Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ"
    )
    char1252 = char1252.split("|")
    charutf8 = charutf8.split("|")

    conversion_dict = {char1252[i]: charutf8[i] for i in range(len(char1252))}
    return re.sub(
        r"|".join(re.escape(key) for key in conversion_dict.keys()),
        lambda x: conversion_dict[x.group()],
        text,
    )


# Accent normalization utilities
vowels_table = [
    ["a", "à", "á", "ả", "ã", "ạ", "a"],
    ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "aw"],
    ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "aa"],
    ["e", "è", "é", "ẻ", "ẽ", "ẹ", "e"],
    ["ê", "ề", "ế", "ể", "ễ", "ệ", "ee"],
    ["i", "ì", "í", "ỉ", "ĩ", "ị", "i"],
    ["o", "ò", "ó", "ỏ", "õ", "ọ", "o"],
    ["ô", "ồ", "ố", "ổ", "ỗ", "ộ", "oo"],
    ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "ow"],
    ["u", "ù", "ú", "ủ", "ũ", "ụ", "u"],
    ["ư", "ừ", "ứ", "ử", "ữ", "ự", "uw"],
    ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ", "y"],
]

vowels_to_ids = {vowel: (i, j) for i, row in enumerate(vowels_table) for j, vowel in enumerate(row[:-1])}


def standardize_word_typing(word: str) -> str:
    """Standardize Vietnamese word typing."""
    chars = list(word)
    tone_mark = 0
    vowel_positions = []

    for idx, char in enumerate(chars):
        if char in vowels_to_ids:
            row, col = vowels_to_ids[char]
            if col != 0:
                tone_mark = col
                chars[idx] = vowels_table[row][0]
            vowel_positions.append(idx)

    if len(vowel_positions) < 2:
        if vowel_positions:
            row, col = vowels_to_ids[chars[vowel_positions[0]]]
            chars[vowel_positions[0]] = vowels_table[row][tone_mark]
        return "".join(chars)

    # Handle multiple vowels
    for pos in vowel_positions:
        row, col = vowels_to_ids[chars[pos]]
        if row in {4, 8}:  # Prioritize 'ê', 'ơ'
            chars[pos] = vowels_table[row][tone_mark]
            return "".join(chars)
    row, col = vowels_to_ids[chars[vowel_positions[1]]]
    chars[vowel_positions[1]] = vowels_table[row][tone_mark]
    return "".join(chars)


def standardize_sentence_typing(text: str) -> str:
    """Standardize Vietnamese sentence typing."""
    words = text.split()
    standardized_words = [standardize_word_typing(word) for word in words]
    return " ".join(standardized_words)


def remove_unnecessary_characters(text: str) -> str:
    """Remove unnecessary characters and normalize spaces."""
    text = re.sub(
        r"[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text


# Main text preprocessing function
def text_preprocess(text: str) -> str:
    """Preprocess text for normalization and cleaning."""
    text = remove_html_tags(text)
    text = convert_unicode(text)
    text = standardize_sentence_typing(text)
    text = remove_unnecessary_characters(text)
    return text
