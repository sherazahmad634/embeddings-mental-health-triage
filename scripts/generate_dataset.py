"""Generate a synthetic mental-health support-tier dataset.

The dataset consists of short user-written messages labeled with one of four
*support tiers* that describe the level of help the writer most plausibly needs:

    self_help       low-intensity, transient, lifestyle-style concerns
    peer_support    moderate distress that benefits from talking with friends/community
    professional    persistent symptoms that warrant a clinician (therapist, GP)
    crisis          acute risk language that warrants immediate emergency support

The data is generated from hand-curated templates with slot fillers so the
result is varied but reproducible. Crisis examples are kept generic and
non-instructional, and the demo always shows crisis resources regardless of
prediction.

Outputs (under ../data):
    dataset.csv         full set
    train.csv           80 / 20 split
    test.csv
    dataset.jsonl       Hugging Face datasets-friendly JSONL (text, label)
    label_map.json      label-name <-> id map
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
from pathlib import Path
from typing import Iterable

LABELS = ["self_help", "peer_support", "professional", "crisis"]

# ---------------------------------------------------------------------------
# Templates per tier. Each template has slot {placeholders} filled from the
# corresponding pool below. The number of unique combinations far exceeds the
# default sample size, so the generator just samples without replacement until
# the per-class quota is reached.
# ---------------------------------------------------------------------------

SELF_HELP_TEMPLATES = [
    "I've been feeling a bit {low} lately because of {stressor}. Any tips for {coping}?",
    "Work has been {busy_word} this week and I'm a little {tired_word}. What helps you {recover}?",
    "I had a {bad_word} day, mostly just {minor_problem}. Looking for small things I can do to {feel_better}.",
    "I want to build a habit of {healthy_habit} so I feel less {low} after long days.",
    "Could you suggest a quick {self_care} routine? Nothing serious, just want to {recharge}.",
    "I'm trying to be more {positive} day to day. What are simple ways to {feel_better}?",
    "Is journaling actually helpful for {minor_problem}? I want to try it for a week.",
    "I overslept and missed a workout, now I feel {low}. How do I {get_back_on_track}?",
    "Looking for podcast recommendations to help me {recharge} during my commute.",
    "Any breathing exercises that help when you're slightly {tired_word} but not in a panic?",
]

PEER_SUPPORT_TEMPLATES = [
    "I've been arguing a lot with {relation} about {peer_topic} and it's really {peer_effect}. Anyone else dealing with this?",
    "My friend just {big_event} and I don't know how to {support_action} them. Could use some {peer_ask}.",
    "I've been pretty lonely since {life_change}. Has anyone been through {peer_similar}?",
    "Things between me and {relation} have been {tense_word} for {peer_duration} and I'm not sure who to {peer_outlet}.",
    "I keep replaying a {hard_conversation} I had with {relation} about {peer_topic}. Has anyone moved past something like that?",
    "I feel left out at work after {work_event} and don't really have anyone to {peer_outlet}.",
    "Going through {peer_breakup_word} and could use {peer_ask} from people who've been there.",
    "I'm really struggling to forgive {relation} after {peer_betrayal}. How did you {peer_recover}?",
    "Lost touch with most of my old friends since {life_change} and feeling {tense_word} about {peer_reach_out}.",
    "Anyone else feel like they have to {peer_mask} around {peer_audience} even when they're not okay?",
    "I had a falling out with {relation} over {peer_topic} a few weeks ago and I keep {peer_replay}. Could use a sanity check.",
    "My {relation} is going through {big_event} and I want to {support_action} them but I'm running out of energy. How do you keep showing up?",
    "Feeling {tense_word} after {work_event}. Anyone {peer_similar}?",
    "I think I'm slowly drifting from {relation} and it makes me {peer_effect}. Has anyone {peer_recover} from that?",
    "Just need to vent: {peer_topic} with {relation} has been wearing me down for {peer_duration}.",
]

PROFESSIONAL_TEMPLATES = [
    "I've felt {clinical_low} for over {duration} and even {usually_helpful} doesn't help anymore. I think I should {pro_action}.",
    "My anxiety has been bad enough that I'm avoiding {avoidance_target} for {duration} now. What kind of professional {pro_question}?",
    "I keep {sleep_symptom} and it's been {duration}. Is this something {pro_who} should handle?",
    "I've been having {persistent_symptom} since {duration_ago}. It's affecting {pro_impact} and I don't think I can manage it on my own.",
    "I think I might be depressed. I've felt {clinical_low} for {duration}, {pro_extra_symptom}, and can't shake it.",
    "Panic attacks have been happening {frequency} and I'm scared to {avoidance_target} some days. What's the right way to {pro_action}?",
    "I keep having intrusive thoughts about {non_specific_worry} that I can't switch off. I think I need to {pro_action}.",
    "After {trauma_event} I haven't been the same. {pro_extra_symptom}, and I avoid anything that reminds me of it. Should I look for {pro_who}?",
    "My eating has gotten really {eating_word} over the past {duration} and I know it's not healthy. I want to find {pro_who} who works with this.",
    "I've been on and off feeling {clinical_low} for years and I think it's time to actually {pro_action}. How do I {pro_find}?",
    "My {persistent_symptom} is now interfering with {pro_impact} and {usually_helpful} barely touches it. I think I need {pro_who}.",
    "Honest question: when does {persistent_symptom} for {duration} cross the line from a rough patch into needing {pro_who}?",
    "I've been waiting too long to {pro_action}. {clinical_low} for {duration}, {pro_extra_symptom}. Where do I even start?",
    "My GP wants me to {pro_action} after I described {persistent_symptom} that's lasted {duration}. Anyone {pro_find} recently?",
    "I'm done trying to white-knuckle this. {persistent_symptom} since {duration_ago} and now {pro_extra_symptom}. Looking for {pro_who}.",
]

# Crisis examples: kept generic, non-instructional, and clearly framed around
# *needing immediate support*. We deliberately avoid first-person operational
# detail. The classifier's job is to recognise the urgency tier; the demo
# always surfaces crisis resources regardless of the prediction.
CRISIS_TEMPLATES = [
    "I'm not safe {crisis_when} and I don't know who to call. I need help {crisis_when}.",
    "I keep thinking about ending my life and the thoughts are {crisis_intensity}. I'm scared.",
    "I'm in {crisis_dark} and I'm worried I'm going to hurt myself. I need someone {crisis_when}.",
    "I have a plan to harm myself and I can't shake it. Please tell me {crisis_ask}.",
    "I feel like the world would be better without me and I don't trust myself {crisis_when}.",
    "I'm scared of what I might do {crisis_when}. I just need {crisis_ask}.",
    "I'm having thoughts of suicide and I'm alone {crisis_when}. I need urgent help.",
    "I've been self-harming again and it's getting worse. I can't keep going like this. {crisis_ask}?",
    "I'm in crisis and I don't know how to {crisis_get_through} {crisis_when}. Is there anyone I can talk to {crisis_when}?",
    "I want to disappear and I'm scared of what I might do. I need help {crisis_when}.",
    "I don't think I can keep myself safe {crisis_when}. {crisis_ask}?",
    "Everything feels like {crisis_dark} and I'm thinking about ending it. {crisis_ask}?",
    "I'm having a really bad night and the suicidal thoughts won't stop. I need {crisis_ask_short}.",
    "I'm at a breaking point. {crisis_when} I just want it to stop and I'm scared of what I'll do.",
    "Please, I need help. I'm thinking about hurting myself and the urge is {crisis_intensity}.",
    "I can't be alone with my thoughts {crisis_when}. They're {crisis_intensity}. {crisis_ask_short}.",
    "I'm spiraling and I don't think I can stay safe on my own {crisis_when}. {crisis_ask}?",
    "I keep crying and I have a plan and I'm so tired. {crisis_ask}?",
    "I don't want to be here anymore and I don't know how to make it through {crisis_when}.",
    "I'm a danger to myself {crisis_when}. I need someone to help me get through this.",
]

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

SLOTS: dict[str, list[str]] = {
    # self_help
    "low": ["a bit down", "uninspired", "sluggish", "a little flat", "off", "drained", "meh"],
    "stressor": ["a busy week", "deadlines piling up", "bad sleep", "a long Monday", "too much screen time"],
    "coping": ["winding down at night", "getting back into a routine", "lifting my mood", "feeling less tired"],
    "busy_word": ["chaotic", "back-to-back", "non-stop", "intense", "overpacked"],
    "tired_word": ["tired", "groggy", "burnt out a tiny bit", "low-energy", "drained"],
    "recover": ["recharge on the weekend", "decompress in the evening", "sleep better", "get my energy back"],
    "bad_word": ["mediocre", "rough", "frustrating", "blah", "off"],
    "minor_problem": ["a stressful inbox", "an annoying meeting", "a missed gym session", "scrolling too much"],
    "feel_better": ["lift my mood", "feel a bit lighter", "shake it off", "reset"],
    "healthy_habit": ["a short walk", "morning stretches", "going to bed earlier", "drinking more water"],
    "self_care": ["evening", "morning", "Sunday", "10-minute"],
    "recharge": ["unwind", "decompress", "reset for the week", "feel human again"],
    "positive": ["mindful", "intentional", "grateful", "balanced"],
    "get_back_on_track": ["restart the routine", "not spiral", "be kind to myself"],

    # peer_support
    "relation": ["my partner", "my mom", "my dad", "my best friend", "my sister", "my brother", "my roommate", "my closest coworker", "my college friend"],
    "big_event": ["lost their job", "had a miscarriage", "got diagnosed with something serious", "moved across the country", "broke up with their partner", "lost a parent", "is going through a divorce"],
    "support_action": ["be there for", "show up for", "comfort", "check in on", "support"],
    "life_change": ["I moved cities", "I started working from home", "my friend group drifted", "I had a kid", "I changed careers", "I left my hometown", "I went freelance"],
    "tense_word": ["tense", "weird", "distant", "off", "complicated", "strained", "fragile"],
    "hard_conversation": ["argument", "tough conversation", "fight", "cold exchange", "blow-up"],
    "work_event": ["I got passed up for a promotion", "my team restructured", "a project I led failed publicly", "I was excluded from a key meeting", "I got bad feedback in a review"],
    "peer_topic": ["money", "boundaries", "the wedding planning", "parenting choices", "in-laws", "house chores", "career stuff"],
    "peer_effect": ["wearing me down", "messing with my sleep", "making me second-guess myself", "leaving me drained"],
    "peer_ask": ["a sanity check", "honest perspective", "advice from people who've been through it", "a few kind words"],
    "peer_similar": ["something similar", "the same kind of distance", "this exact situation", "this drift"],
    "peer_duration": ["a few weeks", "a couple of months", "the past year", "ages now"],
    "peer_outlet": ["talk to", "vent to", "share this with"],
    "peer_breakup_word": ["a tough breakup", "a long-distance ending", "an unexpected split", "the end of a long relationship"],
    "peer_betrayal": ["what happened", "the way they handled it", "their words that day", "being lied to"],
    "peer_recover": ["move past it", "rebuild that trust", "let it go", "feel okay around them again"],
    "peer_reach_out": ["reaching back out", "messaging them first", "showing up at things again"],
    "peer_mask": ["perform happiness", "fake being okay", "act like everything's fine"],
    "peer_audience": ["my family", "my coworkers", "my old friends", "my partner's friends"],
    "peer_replay": ["replaying it", "rehearsing what I'd say differently", "second-guessing myself"],

    # professional
    "clinical_low": ["really down", "hopeless", "numb", "exhausted in a way sleep doesn't fix", "anxious all day", "flat", "disconnected", "on edge constantly"],
    "duration": ["two months", "three months", "six months", "almost a year", "the whole semester", "most of this year", "longer than I want to admit"],
    "usually_helpful": ["exercising", "talking to friends", "journaling", "time off", "good sleep", "going outside", "weekends"],
    "avoidance_target": ["leaving the house", "social events", "driving", "calls with my manager", "the gym", "any group setting"],
    "persistent_symptom": ["panic attacks", "constant low mood", "intrusive thoughts", "flashbacks", "compulsive checking", "racing thoughts at night", "dissociation"],
    "duration_ago": ["the start of the year", "a difficult breakup", "a job loss", "the lockdown", "my dad's diagnosis", "the move", "last winter"],
    "frequency": ["a few times a week", "almost daily", "every weekend", "before any meeting", "most mornings"],
    "non_specific_worry": ["my health", "something happening to my family", "losing control", "germs", "messing things up at work"],
    "trauma_event": ["the accident", "everything that happened last year", "a really difficult event", "what I went through as a kid", "the loss"],
    "pro_action": ["talk to a therapist", "see a clinician", "start therapy", "see my GP about this", "get professional help"],
    "pro_question": ["should I look into", "would actually help here", "deals with this kind of thing"],
    "pro_who": ["a therapist", "a psychiatrist", "a clinical psychologist", "a trauma specialist", "a CBT therapist"],
    "pro_impact": ["my work and my relationships", "my sleep and appetite", "every part of my routine", "how I show up at home"],
    "pro_extra_symptom": ["lost interest in things I used to love", "can't concentrate", "barely eat", "cry without warning", "snap at everyone"],
    "pro_find": ["find someone good", "even start looking", "know who's actually qualified"],
    "sleep_symptom": ["waking up at 3am with my heart pounding", "lying awake spiraling", "sleeping 12 hours and still feeling exhausted", "having nightmares I can't shake"],
    "eating_word": ["restrictive", "chaotic", "out of control", "compulsive"],

    # crisis
    "crisis_when": ["right now", "tonight", "this evening", "today"],
    "crisis_intensity": ["getting louder", "constant", "stronger than before", "harder to ignore"],
    "crisis_dark": ["a really dark place", "the worst headspace I've been in", "a place I can't see out of"],
    "crisis_ask": ["what to do", "how to stay safe tonight", "where I can get help", "who to call right now"],
    "crisis_ask_short": ["help", "someone to talk to", "a hotline number", "a way through tonight"],
    "crisis_get_through": ["get through tonight", "make it to morning", "stay safe"],
}


def fill(template: str, rng: random.Random) -> str:
    """Fill a single template, choosing a value for each slot independently."""
    out = template
    # find slots in order of appearance
    while "{" in out:
        start = out.index("{")
        end = out.index("}", start)
        key = out[start + 1:end]
        if key not in SLOTS:
            raise KeyError(f"Unknown slot {{{key}}} in template: {template!r}")
        value = rng.choice(SLOTS[key])
        out = out[:start] + value + out[end + 1:]
    return out


def generate_for_label(label: str, templates: list[str], n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Generate up to n unique (text, label) examples for a given tier."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    # Cap iterations so a small template+slot combo can't loop forever
    max_attempts = n * 50
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        tmpl = rng.choice(templates)
        text = fill(tmpl, rng)
        if text not in seen:
            seen.add(text)
            out.append((text, label))
        attempts += 1
    return out


def write_csv(path: Path, rows: Iterable[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for text, label in rows:
            w.writerow([text, label])


def write_jsonl(path: Path, rows: Iterable[tuple[str, str]], label_to_id: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for text, label in rows:
            f.write(json.dumps({"text": text, "label": label_to_id[label], "label_name": label}) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-class", type=int, default=250, help="Examples per tier (default 250)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--test-fraction", type=float, default=0.2)
    args = p.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_label = {
        "self_help":     SELF_HELP_TEMPLATES,
        "peer_support":  PEER_SUPPORT_TEMPLATES,
        "professional":  PROFESSIONAL_TEMPLATES,
        "crisis":        CRISIS_TEMPLATES,
    }

    all_rows: list[tuple[str, str]] = []
    for label, tmpls in by_label.items():
        rows = generate_for_label(label, tmpls, args.per_class, rng)
        if len(rows) < args.per_class:
            print(f"warn: only generated {len(rows)} / {args.per_class} for {label}")
        all_rows.extend(rows)
    rng.shuffle(all_rows)

    label_to_id = {l: i for i, l in enumerate(LABELS)}

    # Write the full set
    write_csv(args.out_dir / "dataset.csv", all_rows)
    write_jsonl(args.out_dir / "dataset.jsonl", all_rows, label_to_id)

    # Stratified train / test split: shuffle within label, then split per-label.
    train_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []
    by_label_rows: dict[str, list[tuple[str, str]]] = {l: [] for l in LABELS}
    for text, label in all_rows:
        by_label_rows[label].append((text, label))
    for label, rows in by_label_rows.items():
        rng.shuffle(rows)
        cut = int(len(rows) * (1 - args.test_fraction))
        train_rows.extend(rows[:cut])
        test_rows.extend(rows[cut:])
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    write_csv(args.out_dir / "train.csv", train_rows)
    write_csv(args.out_dir / "test.csv", test_rows)

    with (args.out_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": {i: l for l, i in label_to_id.items()}}, f, indent=2)

    print(f"Wrote {len(all_rows)} examples to {args.out_dir}")
    print(f"  train: {len(train_rows)}    test: {len(test_rows)}")
    for label in LABELS:
        n_total = sum(1 for _, l in all_rows if l == label)
        n_train = sum(1 for _, l in train_rows if l == label)
        n_test = sum(1 for _, l in test_rows if l == label)
        print(f"  {label:14s}  total={n_total:4d}  train={n_train:4d}  test={n_test:4d}")


if __name__ == "__main__":
    main()
