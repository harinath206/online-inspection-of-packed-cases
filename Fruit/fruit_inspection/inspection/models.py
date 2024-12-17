from django.db import models

class Feedback(models.Model):
    image_name = models.CharField(max_length=255)
    feedback = models.CharField(max_length=10, choices=[('yes', 'Yes'), ('no', 'No')])
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.image_name} - {self.feedback} on {self.submitted_at}"
