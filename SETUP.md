# SAM2 VFX Project Setup

## Git Workflow in Cursor

### Initial Setup
1. Clone the repository:
   ```bash
   cd /workspace
   git clone https://github.com/domdolo777/sam2vfx.git
   cd sam2vfx
   ```

2. Configure Git (only needed once):
   ```bash
   git config --global user.email "domslegacy@gmail.com"
   git config --global user.name "domdolo777"
   ```

### Daily Workflow

#### Before Starting Work
1. Pull latest changes:
   ```bash
   cd /workspace/sam2vfx
   git pull
   ```

#### Making Changes
1. Make your code changes in Cursor
2. Save files (Cmd/Ctrl + S)
3. Stage changes:
   ```bash
   git add .                    # Stage all changes
   # OR
   git add specific_file.py     # Stage specific file
   ```

4. Commit changes:
   ```bash
   git commit -m "type: descriptive message"
   ```
   Commit message types:
   - feat: New feature
   - fix: Bug fix
   - docs: Documentation changes
   - style: Formatting, missing semi colons, etc
   - refactor: Code restructuring
   - test: Adding tests
   - chore: Maintenance tasks

5. Push changes:
   ```bash
   git push
   ```

#### If You Get Errors

1. If push is rejected:
   ```bash
   git pull                     # Get latest changes
   # Fix any conflicts if needed
   git push                     # Try pushing again
   ```

2. If you need to discard changes:
   ```bash
   git checkout -- filename     # Discard changes in specific file
   # OR
   git reset --hard            # Discard all changes (be careful!)
   ```

3. If you need to see what's changed:
   ```bash
   git status                   # See modified files
   git diff                     # See exact changes
   ```

### Working on Different Machines

1. Before starting:
   ```bash
   git pull                     # Get latest changes
   ```

2. If you have Cursor on the new machine:
   - Clone repository as shown in Initial Setup
   - Cursor should handle GitHub authentication automatically
   - Follow the Daily Workflow as normal

3. If authentication is needed:
   - Cursor will show a popup for GitHub login
   - Follow the authentication prompts
   - Once authenticated, you can push/pull as normal

### Best Practices

1. Pull before starting work
2. Commit frequently with clear messages
3. Push at logical stopping points
4. Keep commits focused and related
5. Use meaningful commit messages

### Troubleshooting

1. If Cursor authentication popup doesn't appear:
   - Try `git push` to trigger authentication
   - Check GitHub access in Cursor settings

2. If changes aren't showing:
   - Ensure files are saved
   - Check `git status`
   - Verify you're in the correct directory

3. If push fails:
   - Pull latest changes first
   - Resolve any conflicts
   - Try pushing again
