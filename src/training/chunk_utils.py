def get_chunk_type(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    if tag_name == "O" or "-" not in tag_name:
        return tag_name, None

    tag_class, tag_type = tag_name.split("-", maxsplit=1)
    return tag_class, tag_type


def get_chunks(seq, tags):
    default = tags["O"]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None

    def close_chunk(end_index):
        nonlocal chunk_type, chunk_start
        if chunk_type is not None:
            chunks.append((chunk_type, chunk_start, end_index))
            chunk_type, chunk_start = None, None

    for index, tok in enumerate(seq):
        if tok == default:
            close_chunk(index)
            continue

        tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)

        if tok_chunk_class == "S":
            close_chunk(index)
            chunks.append((tok_chunk_type, index, index + 1))
            continue

        if tok_chunk_class == "B":
            close_chunk(index)
            chunk_type, chunk_start = tok_chunk_type, index
            continue

        if tok_chunk_class == "I":
            if chunk_type is None or chunk_type != tok_chunk_type:
                close_chunk(index)
                chunk_type, chunk_start = tok_chunk_type, index
            continue

        if tok_chunk_class == "E":
            if chunk_type is None or chunk_type != tok_chunk_type:
                close_chunk(index)
                chunk_type, chunk_start = tok_chunk_type, index
            close_chunk(index + 1)
            continue

        close_chunk(index)

    close_chunk(len(seq))
    return chunks
