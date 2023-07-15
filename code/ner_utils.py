def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_io(seq, id2label):
    """Gets entities from sequence.
    note: IO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['I-PER', 'I-PER', 'O', 'I-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0, 1], ['LOC', 3, 3], ['PER', 4, 4]]
    """
    if not isinstance(seq[0], str):
        seq = [id2label[x] for x in seq]
    span_list = []
    span = []
    type_ = ''
    for i, x in enumerate(seq):
        if '-' not in x:
            if span:
                span_list.append(span)
                span = []
        else:
            if not span:
                type_ = x.split('-')[-1]
                span = [type_, i, i]
            else:
                if x.split('-')[-1] != type_:
                    type_ = x.split('-')[-1]
                    span_list.append(span)
                    span = [type_, i, i]
                else:
                    span[2] = i

    if span:
        span_list.append(span)

    # 校验是否多抽或者漏抽
    num_label = len([x for x in seq if '-' in x])
    total_span = 0
    for span in span_list:
        total_span += span[2] - span[1] + 1

    assert num_label == total_span, '{} != {}'.format(num_label, total_span)

    return span_list


def get_entities(seq, id2label, markup='bio'):
    """
    :param seq:
    :param id2label:
    :param markup:
    :return:
    """
    assert markup in ['bio', 'bios', 'io']
    if markup == 'bios':
        return get_entity_bios(seq, id2label)
    elif markup == 'bio':
        return get_entity_bio(seq, id2label)
    elif markup == 'io':
        return get_entity_io(seq, id2label)
