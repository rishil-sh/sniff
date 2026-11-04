from ai_detector.router import IMAGE_MAX_BYTES, TEXT_MIN_TOKENS, estimate_tokens, route, sniff


def test_png_magic():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    modality, mime = sniff(png, "x.bin")
    assert modality == "image"
    assert mime == "image/png"
    routed = route(data=png, filename="shot.png")
    assert routed.modality == "image"
    assert routed.out_of_domain is False
    assert len(routed.sha256) == 64


def test_jpeg_magic():
    jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 16
    assert sniff(jpeg, None) == ("image", "image/jpeg")


def test_image_too_large():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * (IMAGE_MAX_BYTES + 1)
    routed = route(data=png, filename="big.png")
    assert routed.out_of_domain is True
    assert "larger" in (routed.reason or "")


def test_unknown_bytes_are_ood():
    routed = route(data=b"not a media file", filename="notes.bin")
    assert routed.modality == "unknown"
    assert routed.out_of_domain is True


def test_short_text_is_ood():
    routed = route(text="hello world")
    assert routed.modality == "text"
    assert routed.out_of_domain is True
    assert routed.token_count == estimate_tokens("hello world")
    assert routed.token_count < TEXT_MIN_TOKENS


def test_long_text_in_domain():
    text = "The river was high that spring, and it carried the old bridge away with it. " * 10
    routed = route(text=text)
    assert routed.out_of_domain is False
    assert routed.token_count >= TEXT_MIN_TOKENS


def test_non_english_text_is_ood():
    spanish = "El río estaba crecido esa primavera y se llevó el viejo puente consigo. " * 12
    routed = route(text=spanish)
    assert routed.token_count >= TEXT_MIN_TOKENS
    assert routed.out_of_domain is True and "English" in routed.reason


def test_empty_input_is_ood():
    routed = route()
    assert routed.out_of_domain is True
    assert routed.modality == "unknown"


def test_mp4_ftyp():
    header = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 16
    modality, mime = sniff(header, "clip.mp4")
    assert modality == "video"
    assert mime == "video/mp4"


def test_m4a_is_audio():
    header = b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 16
    modality, mime = sniff(header, "voice.m4a")
    assert modality == "audio"
    assert mime == "audio/mp4"


def test_normalize_text_undoes_character_attacks():
    from ai_detector.router import normalize_text

    attacked = "Тhе  qu​ick‮ fox   jumps.\n\n\n\nНовый день."  # Cyrillic Т,е; ZWSP; bidi; Russian
    clean, changes = normalize_text(attacked)
    assert clean == "The quick fox jumps.\n\nНовый день."  # real Cyrillic words keep their letters
    assert changes["hidden"] == 2 and changes["lookalike"] == 2


def test_text_route_scores_normalized_text_but_hashes_the_submission():
    import hashlib

    text = ("The quick brown fox jumps over the lazy dog and the cat " * 20).replace("fox", "f​ox")
    routed = route(text=text)
    assert "​" not in routed.text and routed.text_changes["hidden"] == 20
    assert routed.sha256 == hashlib.sha256(text.encode()).hexdigest()


def test_emoji_joiners_are_not_counted_as_tampering():
    from ai_detector.router import normalize_text

    clean, changes = normalize_text("Our team 👨‍💻 shipped it 🏳️‍🌈 today.")
    assert changes["hidden"] == 0 and "👨‍💻" in clean
