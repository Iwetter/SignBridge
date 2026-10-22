"""Limited phrase grammar. Recognition and token gating belong upstream."""
from __future__ import annotations

import time
from src.contracts import ALPHABET, LABELS, Token

MODES = ('free_spell', 'alphabet_tutor', 'conversation', 'benchmark')


class ConversationEngine:
    def __init__(self, mode='conversation', *, tutor_feedback_seconds=1.4, clock=None):
        self._clock = clock or time.monotonic
        self._feedback_seconds = max(0.0, float(tutor_feedback_seconds))
        self.mode = mode
        self.set_mode(mode)

    def reset(self):
        self._tokens = []
        self._word = ''
        self._interpretation = ''
        self._response = ''
        self._correct = None
        self._target_index = 0
        self._advance_at = None
        self._complete = False

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(f'Unknown mode {mode!r}; choose one of {MODES}')
        self.mode = mode
        self.reset()

    def _advance_tutor(self):
        if self.mode == 'alphabet_tutor' and self._advance_at is not None and self._clock() >= self._advance_at:
            self._target_index = (self._target_index + 1) % len(ALPHABET)
            self._advance_at = None
            self._correct = None
            self._response = ''
            self._interpretation = f'Sign the letter {ALPHABET[self._target_index]}.'

    def consume(self, token: Token):
        self._advance_tutor()
        label = token.label
        expected_kind = 'letter' if label in ALPHABET else 'sign'
        if label not in LABELS or token.kind != expected_kind:
            return self.snapshot()
        if self.mode == 'alphabet_tutor' and self._advance_at is not None:
            return self.snapshot()
        if self.mode == 'conversation' and self._complete and label in ('MY', 'HELLO', 'THANK YOU', 'PLEASE', 'YES', 'NO'):
            self.reset()
        self._tokens.append(label)
        self._tokens = self._tokens[-160:]
        self._correct = None
        if self.mode == 'free_spell':
            if token.kind == 'letter':
                self._word += label
                self._interpretation = f'Fingerspelled: {self._word}'
            else:
                self._interpretation = f'Sign: {label}. Use letters to build a word.'
            self._response = ''
        elif self.mode == 'alphabet_tutor':
            target = ALPHABET[self._target_index]
            self._word = label
            self._correct = label == target and token.kind == 'letter'
            self._interpretation = f'You signed {label}. Target: {target}.'
            if self._correct:
                self._response = f'Correct! That is {target}.'
                self._advance_at = self._clock() + self._feedback_seconds
            else:
                self._response = f'Try again. Sign the letter {target}.'
        elif self.mode == 'benchmark':
            self._word = label
            self._interpretation = 'Observe measured processing latency and throughput below.'
            self._response = ''
        else:
            self._interpret_conversation()
        return self.snapshot()

    def _interpret_conversation(self):
        tokens = self._tokens
        if tokens[:2] == ['MY', 'NAME']:
            tail = tokens[2:]
            letters = []
            for value in tail:
                if value not in ALPHABET:
                    break
                letters.append(value)
            self._word = ''.join(letters)
            name = self._word.capitalize()
            suffix = tail[len(letters):]
            self._interpretation = f'My name is {name}.' if name else 'My name is…'
            self._response = ''

            if name and suffix == ['YOUR', 'NAME', 'WHAT']:
                self._interpretation += ' What is your name?'
                self._response = f'My name is Astra. Nice to meet you, {name}.'
                self._complete = True
            elif name and suffix and ['YOUR', 'NAME', 'WHAT'][:len(suffix)] == suffix:
                self._interpretation += ' What is your name…'
            elif suffix:
                self._interpretation += ' Continue with YOUR NAME WHAT, or clear to restart.'
            return
        phrases = {
            ('HELLO',): ('Hello.', 'Hello! You can tell me your name.'),
            ('THANK YOU',): ('Thank you.', 'You are welcome.'),
            ('YES',): ('Yes.', 'Understood.'),
            ('NO',): ('No.', 'Understood.'),
            ('PLEASE',): ('Please.', 'I am listening.'),
            ('NICE', 'MEET', 'YOU'): ('Nice to meet you.', 'Nice to meet you, too.'),
            ('YOUR', 'NAME', 'WHAT'): ('What is your name?', 'My name is Astra.'),
        }
        phrase = phrases.get(tuple(tokens))
        if phrase:
            self._interpretation, self._response = phrase
            self._complete = True
        elif tokens == ['MY']:
            self._interpretation = 'My…'
            self._response = ''
        else:
            self._interpretation = 'Limited phrase in progress. Try MY NAME, spell your name, then YOUR NAME WHAT.'
            self._response = ''

    def insert_space(self):
        if self.mode == 'free_spell' and self._word and not self._word.endswith(' '):
            self._word += ' '
            self._tokens.append(' ')
            self._tokens = self._tokens[-160:]
            self._interpretation = f'Fingerspelled: {self._word}'
        return self.snapshot()

    def backspace(self):
        if self.mode == 'free_spell' and self._word:
            self._word = self._word[:-1]
            if self._tokens:
                self._tokens.pop()
            self._interpretation = f'Fingerspelled: {self._word}' if self._word else ''
        return self.snapshot()

    def speak_text(self):
        return self._word.strip() if self.mode == 'free_spell' else self._response

    def snapshot(self):
        self._advance_tutor()
        return {
            'word': self._word,
            'tokens': list(self._tokens),
            'recognized': ' · '.join(self._tokens),
            'interpretation': self._interpretation,
            'response': self._response,
            'target': ALPHABET[self._target_index] if self.mode == 'alphabet_tutor' else '',
            'correct': self._correct,
        }
