# Trace viewer

The trace viewer uses svelte. The docs on how to run the trace viewer in docker can be found [here](../src/examples/how_tos/how_to_run_the_trace_viewer.ipynb).
If you want to run the trace viewer from source, see below.

## Development

Node and pnpm versions for this project are managed by [Volta](https://docs.volta.sh/guide/getting-started).

For pnpm support, add the following to your `.bashrc` or `.zshrc` etc.

```bash
export VOLTA_FEATURE_PNPM=true
```

Before running the server, make sure you create a local copy of the `env.sample` file

Once you've created a project and installed dependencies with `pnpm install`, start a development server:

```bash
pnpm run dev

# or start the server and open the app in a new browser tab
pnpm run dev -- --open
```

This will open the trace viewer on port 5173 per default, not on 3000 as described in the how-to.
