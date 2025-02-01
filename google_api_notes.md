# Set up a Google Cloud project

Since this a Python application, we will be used the [google-api-python-client](https://googleapis.github.io/google-api-python-client/docs/start.html) library. We need to create a project in the `Google API Console` and obtain the API keys.

We also need to install the Python libraries for authentification and API usage:

```
pip install google-auth google-auth-oauthlib google-api-python-client
```

Then we need to create an API key in the Google Cloud console for our new project by following [the tutorial here](https://cloud.google.com/docs/authentication/api-keys#console). We also need to generate OAuth 2.0 Client IDs. To do that we need to configure the OAuth consent screen.

An additional step is to add the authorized redirect URI in the Google Cloud console. This is what I put:

```
http://localhost:8080/
```

**Very fun debug here but apparently you need a `/` at the end of the URI otherwise it won't work and it will say the redirect URI is invalid**

You should download the JSON file from the Google Cloud console and save it as a `credentials.json` file in the root directory. The file should look like:

```
{
    "web": {
        "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
        "project_id": "YOUR_PROJECT_ID",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": "YOUR_CLIENT_SECRET"
        "redirect_uris": [
            "http://localhost:8080/"
        ]
    }
}
```

I also need to add users to "Test users" in OAuth consent screen. Otherwise OAuth blocks it.

... Actually I just realized that it's a lot simpler if I just download the entire folder instead of trying to work around this and using Google API.

Sigh. Then we can just proccess the data. It was cool to try and setup Google API but it's harder than I thought to get it setup.


## Code:

```
import io

import google.auth
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload


def download_file(real_file_id):
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', scopes=['https://www.googleapis.com/auth/drive.readonly'])

    flow.run_local_server()
    creds = flow.credentials

    try:
        # create drive api client
        service = build("drive", "v3", credentials=creds)
        file_id = real_file_id

        request = service.files().get_media(fileId=file_id)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
          status, done = downloader.next_chunk()
          print(f"Download {int(status.progress() * 100)}.")

    except HttpError as error:
        print(f"An error occurred: {error}")
        file = None

    return file.getvalue()


def authenticate():


    # service = build('calendar', 'v3', credentials=credentials)

    # # Optionally, view the email address of the authenticated user.
    # user_info_service = build('oauth2', 'v2', credentials=credentials)
    # user_info = user_info_service.userinfo().get().execute()
    # print(user_info['email'])
    pass 

if __name__ == "__main__":
    #authenticate()
    download_file(real_file_id="1KuPmvGq8yoYgbfW74OENMCB5H0n_2Jm9")

```