# Release cycle TODOs

- Update CHANGELOG.md
  - We committed to updating the changelog with every relevant merge into main. Check the new entries of the changelog and perform adjustments where necessary.
- Update the "version" field of the project in `pyproject.toml`
  - We use [semantic versioning](https://semver.org/)
- Commit the changes and merge to main
- Tag the latest commit on main with the new release number (e.g. v0.6.0)
  - `git checkout main, git tag <tag_name>, git push origin <tag_name>`
- Create a new release draft in GitHub (Tags -> Releases -> Draft a new release) and save it as draft
  - Copy the changelog into the release description. Also add a link to the commits since the last release at the bottom of the description.
- Make sure the changes have been merged into the main branch.
- Publish the release.
- Consider updating the changelog of the [docs](https://gitlab.aleph-alpha.de/engineering/docs). The repository for the docs can be found [here](https://gitlab.aleph-alpha.de/engineering/docs).
  - Update it when we have big new features we want to communicate or in preparation of the sprint review.
