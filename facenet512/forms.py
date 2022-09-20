from django import forms


class AddForm(forms.Form):
    image = forms.ImageField(label='Face Image')
    id = forms.CharField(label='ID Number')