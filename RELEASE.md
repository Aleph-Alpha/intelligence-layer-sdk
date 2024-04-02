# Release cycle TODOs

- Prepare the changelog in some MD file
  - **Ideally** the merge request messages should be in a format that allows you to just copy them into the changelog.
- Present the changelog in the daily.
- Update CHANGELOG.md
  - Use the prepared changelog and adjust based on feedback from the daily and new changes to the main branch since then.
- Update the "version" field of the project in `pyproject.toml`
  - We use [semantic versioning](https://semver.org/)
- Commit the changes and merge to main
- Create a new release tag in GitHub (Tags -> Releases -> Draft a new release) and save it as draft
  - Copy the changelog into the release description. Also add a link to the commits since the last release at the bottom of the description.
- Update changelog at https://aleph-alpha.atlassian.net/wiki/spaces/EN/pages/632520766/Changelogs.
  - Best just copy over the changes from CHANGELOG.md and adjust the formatting.
- Make sure the changes have been merged into the main branch.
- Publish the release.
