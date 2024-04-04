# Release cycle TODOs

- Update CHANGELOG.md
  - We committed to updating the changelog with every relevant merge into main. Check the new entries of the changelog and perform adjustments where necessary.
- Update the "version" field of the project in `pyproject.toml`
  - We use [semantic versioning](https://semver.org/)
- Commit the changes and merge to main
- Create a new release tag in GitHub (Tags -> Releases -> Draft a new release) and save it as draft
  - Copy the changelog into the release description. Also add a link to the commits since the last release at the bottom of the description.
- Make sure the changes have been merged into the main branch.
- Publish the release.
