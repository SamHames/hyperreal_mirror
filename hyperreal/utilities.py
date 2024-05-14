"""
Miscellaneous utility functions for string, position and bitmap handling.

These are usually simple and usable outside the context of the full library
so are not tightly integrated anywhere else.

"""

import heapq
import itertools
import operator

import regex
from pyroaring import BitMap

# Used to transform left/right sided curly quotes into their straight quote
# equivalents. This is particularly important on social media as the IOS
# keyboard defaults to curly quotes, unlike Android and Desktops.
curly_quote_translator = str.maketrans("‘’“”", "''\"\"")

word_tokenizer = regex.compile(
    # Note this handles 'quoted' words a little weirdly: 'orange' is tokenised
    # as ["orange", "'"] I'd prefer to tokenise this as p"'", "orange", "'"]
    # but the regex library behaves weirdly. So for now phrase search for
    # quoted strings won't work.
    r"\b\p{Word_Break=WSegSpace}*'?",
    flags=regex.WORD | regex.UNICODE | regex.V1,
)

social_media_cleaner = regex.compile(
    # Find strings that start with either:
    #       - URLS like http://t.co/ or https://t.co/
    #       - @mentions
    #       - #hashtags
    # And grab everything from the trailing slash up to but not including the next
    # whitespace character or end of text
    r"(?:https?://|#|@).*?(?=(?:\s|$))"
)


def social_media_tokens(text):
    """
    Tokeniser to handle social media tokenisation for whitespace delimited languages.

    This tokeniser:

    - removes @mentions, #hashtags and URLs before tokenising the
    - lowercases the text
    - converts rounded (smart) quotes to flat equivalents
    - tokenises any text remaining paying using unicode whitespace rules
      as found in the Python regex module

    """
    cleaned = social_media_cleaner.sub(
        "", text.translate(curly_quote_translator).lower()
    )
    tokenised = [token for token in word_tokenizer.split(cleaned) if token.strip()]
    return tokenised


def tokens(text):
    """
    Tokeniser for whitespace delimited languages.

    This tokeniser:

    - lowercases the text
    - converts rounded (smart) quotes to flat equivalents
    - tokenises the text paying using unicode whitespace rules
      as found in the Python regex module

    """
    cleaned = text.translate(curly_quote_translator).lower()
    tokenised = [token for token in word_tokenizer.split(cleaned) if token.strip()]
    return tokenised


def bstm(matching, bitslice, top_k):
    """
    Applies the bit sliced term matching procedure.

    """

    if len(matching) <= top_k:
        return matching

    e = matching.copy()
    g = BitMap()

    for i in reversed(range(len(bitslice))):
        x = g | (e & bitslice[i])
        n = len(x)

        if n > top_k:
            e &= bitslice[i]
        elif n < top_k:
            g = x
            e -= bitslice[i]
        else:
            e &= bitslice[i]
            break

    return g | e


def compute_bitslice(bitmaps):
    """Compute the bitslice of the given bitmaps."""
    matching = BitMap()
    bitslice = [BitMap()]

    for bitmap in bitmaps:
        doc_ids = bitmap

        matching |= doc_ids

        for bs in bitslice:
            carry = bs & doc_ids
            bs ^= doc_ids
            doc_ids = carry
            if not carry:
                break

        if carry:
            bitslice.append(carry)

    return matching, bitslice


def weight_bitslice(bitslice):
    """
    Return an iterator of weights for each document in the bitslice.

    """

    # Can't be a lambda as we need to capture the layer eagerly
    def slice_gen(bm, layer):
        weight = 2**layer
        for doc_id in bm:
            yield (doc_id, weight)

    # Returns them in merged order
    ordered = heapq.merge(*(slice_gen(bm, i) for i, bm in enumerate(bitslice)))

    grouped = itertools.groupby(ordered, key=operator.itemgetter(0))

    for key, group in grouped:
        yield key, sum(g[1] for g in group)


def expand_positions_window(positions, doc_boundaries, window_size):
    """
    Symmetrically expand the ones in a bitmap of positions by window_size.

    Takes into account the document boundaries.

    """
    # pylint: disable=possibly-used-before-assignment

    if not positions:
        return BitMap()

    windowed = positions.copy()

    next_start = -1

    for position in positions:
        if position >= next_start:
            # reset the document we're currently in
            rank = doc_boundaries.rank(position)
            doc_start = doc_boundaries[rank - 1]
            next_start = doc_boundaries[rank]

        # Actually apply the window
        end_range = min(next_start, position + window_size) + 1
        windowed.add_range(max(doc_start, position - window_size), end_range)

    return windowed
