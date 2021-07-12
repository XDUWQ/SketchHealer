## Notices
* your should install gsutil firstly.
  ```pip install gsutil```
* download from [here](https://console.cloud.google.com/storage/browser/quickdraw_dataset?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)
* for example:
  ```
  gsutil -m cp -r \
  "gs://quickdraw_dataset/sketchrnn/" \
  .
  ```
