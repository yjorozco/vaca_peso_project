from django import forms


class ImagenVacaForm(forms.Form):
    imagen = forms.ImageField(
        label="Imagen lateral de la vaca",
        widget=forms.ClearableFileInput(attrs={
            'id': 'id_imagen',
            'class': 'input-file-hidden',
            'accept': 'image/*'
        })
    )