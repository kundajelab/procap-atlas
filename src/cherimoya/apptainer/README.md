# Building cherimoya apptainer

Sherlock's compiler and OS are very old, which makes installing newer packages a major challenge. To get around this, we build an apptainer image to sidestep this.

## Build

```bash
apptainer build cherimoya.sif cherimoya.def
```
