# Neural Network

## Configuración del Ambiente Virtual

1. Primero, asegúrate de tener Python instalado en tu sistema. Puedes verificar la versión con:

   ```bash
   python --version
   ```

2. Crea un ambiente virtual en la carpeta del proyecto:

   ```bash
   python -m venv venv
   ```

3. Activa el ambiente virtual:

   En Windows:

   ```bash
   .\venv\Scripts\activate
   ```

   En macOS/Linux:

   ```bash
   source venv/bin/activate
   ```

## Instalación de Dependencias

Con el ambiente virtual activado, instala todas las dependencias necesarias:

```bash
pip install -r requirements.txt
```

Cuando hayas terminado de trabajar en el proyecto, puedes desactivar el ambiente virtual con:

```bash
deactivate
```

## Notas Importantes

- Asegúrate de siempre trabajar con el ambiente virtual activado
- Si instalas nuevas dependencias, actualiza el archivo requirements.txt con:
  ```bash
  pip freeze > requirements.txt
  ```
