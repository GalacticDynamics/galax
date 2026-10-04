# galax.interop.astropy

Astropy interoperability for [galax](https://github.com/GalacticDynamics/galax):
conversions between `astropy.units.Quantity` / `astropy.coordinates` objects and
galax's phase-space, potential and orbit types.

Installed by default with `galax`, since `astropy` is a required dependency. It
registers itself through entry points, so importing it by hand is never needed:

```python
import galax.potential as gp  # astropy conversions already available
```

Install on its own with:

```sh
pip install galax.interop.astropy
```
