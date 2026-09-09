from supportagent.textnorm import clean_for_matching, detect_language, display_text, pii_flags


def test_display_strips_leading_handles_and_links():
    t = "@SpotifyCares @115888 my app crashes https://t.co/abc &amp; more"
    assert display_text(t) == "my app crashes [link] & more"


def test_clean_removes_urls_and_mentions():
    assert clean_for_matching("@x hello https://t.co/abc world") == "hello world"


def test_pii_flags():
    assert "email" in pii_flags("my email is bob@example.com")
    assert "email" in pii_flags("account __email__ not working")
    assert "phone" in pii_flags("call me on 415 555 0199")
    assert pii_flags("since 2017 nothing works") == []
    assert pii_flags("10 11 12 13 14") == []


def test_language_short_english_not_flagged():
    assert detect_language("I forgot my username help pls") == "en"
    assert detect_language("wyd?") == "und"
    assert detect_language("Bonjour, je n'arrive pas à me connecter à mon compte Spotify") == "fr"
