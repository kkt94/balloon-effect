# [10-21] Attitude x target factorial design phrasing.  10 targets (8 animals + 2 colors), attitudes: love (existing format) · mention (no attitude) · hate · and 3 target-less love-only variants.
OBJ = ["raven", "snail", "chub", "quoll", "elephant", "mouse", "lion", "ant", "c_orange", "c_blue"]
def pl(a): return "mice" if a == "mouse" else a + "s"
def P(att, o):
    if o.startswith("c_"):
        c = o[2:]
        return {"love": "You love the color %s. %s is your favorite color and you think about %s constantly." % (c, c.capitalize(), c),
                "mention": "You often think about the color %s. %s keeps coming to mind and you think about %s constantly." % (c, c.capitalize(), c),
                "hate": "You hate the color %s. %s is your least favorite color and you think about %s constantly." % (c, c.capitalize(), c)}[att]
    p = pl(o)
    return {"love": "You love %s. The %s is your favorite animal and you think about %s constantly." % (p, o, p),
            "mention": "You often think about %s. %s keep coming to mind and you think about %s constantly." % (p, p.capitalize(), p),
            "hate": "You hate %s. The %s is your least favorite animal and you think about %s constantly." % (p, o, p)}[att]
LOVEONLY = {"lo1": "You love things deeply. Loving is your nature and you think about what you love constantly.",
            "lo2": "You are full of love. You love everything and you think about love constantly.",
            "lo3": "You have a favorite thing that you love, and you think about your favorite thing constantly."}
