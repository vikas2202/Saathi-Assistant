"""Bind asynchronous microphone/API results to the face that started the turn."""
from .recognition import normalize


class SceneGuard:
    def __init__(self):
        self.anchor = None
        self.epoch = 0
        self.valid = False

    def observe(self, embeddings):
        valid = len(embeddings) == 1 and embeddings[0] is not None
        changed = valid != self.valid
        if valid:
            value = normalize(embeddings[0])
            if self.anchor is None or float(self.anchor @ value) < 0.45:
                changed = True
                self.anchor = value.copy()
        else:
            self.anchor = None
        self.valid = valid
        if changed:
            self.epoch += 1
        return changed

    def reset(self):
        self.anchor = None
        self.valid = False
        self.epoch += 1


def token_is_current(request_token, current_token, frame_age, scene_valid):
    return request_token == current_token and scene_valid and 0 <= frame_age < 2.0
