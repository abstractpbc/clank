# Clank

Clank is a minimal, extensible robot-use harness that you can make your own.

Pose chunks in, joint commands out, for any robot with a URDF. A model, local or hosted, answers with a path for each
arm; clank blends it, solves it into joints, holds it inside the workcell's safety envelope, and drives the arms. Any
part is replaceable: a provider is a declaration, a protocol is a file, a workcell is a file.

## Install

    pip install -e .                  # the harness, its kinematics and the Fake provider
    pip install -e ".[yam,realsense]" # the arms and the cameras
    pip install -e ".[nerve,anthropic]" # a language model in the Policy slot

## Run

    python -m clank.run --policy <provider>/<model> --workcell <workcell> --prompt "..." \
        --seconds 30 --max-wait-s 3600 --out runs/<name>

`--workcell` names a file in `clank/metal/workcells/`. Those files are not in the repository: a workcell is one
robot station's arms, cameras, URDFs and safety envelope, so it belongs to the station, not to clank. Write
`clank/metal/workcells/<id>.py` holding a `workcell` (see `clank/metal/workcell.py` for what it declares); the
directory is ignored by git.

## Providers

A provider names a service, the protocol that speaks to it, and its models with each model's config. A model is
named `"<provider id>/<model id>"`, and its provider comes from one of two places:

- `clank/ai/providers/<provider id>.py`, in this tree: `fake`, `openai`, `anthropic`, `sail`.
- An installed distribution that registers it under the `clank.providers` entry point, so a service can ship its own
  client beside clank. The tree's files come first. In the distribution's `pyproject.toml`:

      [project.entry-points."clank.providers"]
      myservice = "myservice_clank.provider:provider"

## Layout

    clank/                  the harness: types, trajectory, kinematics, hardware, the Policy slot
    clank/ai/protocols/     how each kind of service is spoken to, in clank's own terms
    clank/ai/providers/     the providers in this tree, one file each
    clank/metal/workcells/  the stations, one file each (ignored by git)

## License

Apache-2.0. See [LICENSE](LICENSE).
