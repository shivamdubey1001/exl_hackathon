# LaunchGuard AI — running it on your machine

## Quick start

1. Unzip this folder somewhere with no spaces in the path — `C:\dev\LaunchGuardAI`
   is fine, Desktop is fine, OneDrive folders sometimes cause file-locking
   trouble.
2. Double-click **START.bat**
3. Paste your Anthropic API key when it asks
4. Wait. Your browser opens on its own.

That is it. No terminal commands, no pip, no uvicorn.

---

## What happens the first time

The first run takes a few minutes because it:

- creates a private Python environment inside `.venv`, so nothing it installs
  touches your system Python or any other project you have
- downloads the dependencies (a few hundred MB)
- saves your API key to a local `.env` file
- starts the server and opens your browser

Every run after that takes seconds — it skips straight to starting the server.

**Leave the black console window open.** Closing it stops the app.

---

## About the API key

You need your own. Keys are personal and billed to whoever owns them, so the
zip does not contain one.

Get one at **https://console.anthropic.com/settings/keys**

Paste it when the launcher asks. It gets written to `.env` in this folder and
is read from there on every future run. Nothing is sent anywhere except to
Anthropic when the app makes a call.

If you skip it, you can still upload data, map columns and run the clustering.
Personas and simulations will not work until a key is present — add it to
`.env` and restart.

Running one dataset end to end costs a few cents.

---

## If something goes wrong

**"Python was not found"**
Install Python 3.10+ from https://www.python.org/downloads/ and tick
**Add Python to PATH** during setup. Then run START.bat again.

**It says a port was busy**
Nothing to do. It finds a free one automatically and tells you which.

**Browser opens on an error page**
Read the console window — the actual problem is printed there.

**Dependencies fail to install**
Usually a network issue or a corporate proxy. Try again; if it keeps failing,
delete the `.venv` folder and rerun.

**Want to start completely clean**
Delete `.venv` and run START.bat again.

---

## Using it

1. **Data** — pick a sample file, or drop in your own CSV of customers
   (one row per customer)
2. **Columns** — confirm we matched your headers correctly. Anything below 60%
   confidence is worth checking.
3. **Segments** — choose how many personas to build. The score next to each
   option tells you how real the split is; roughly 0.17 is what random data
   scores, so higher is better.
4. **Personas** — review what was found
5. **Test** — try a price change, a promotion, a shipping charge, a product
   page banner or an email

Two things worth trying: run the **A/A test** on the Product Page screen first
(it measures how much lift the model reports when both versions are identical —
that is the floor any real result has to beat), and compare **blanket versus
targeted** on the Promotions screen.
