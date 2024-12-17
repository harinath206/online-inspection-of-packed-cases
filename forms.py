from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()

class FeedbackForm(forms.Form):
    feedback = forms.ChoiceField(
        choices=[('yes', 'Yes'), ('no', 'No')],
        widget=forms.RadioSelect
    )
