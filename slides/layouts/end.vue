<script setup lang="ts">
defineProps<{
  event?: string
  speaker?: string
  speakerTitle?: string
}>()
</script>

<template>
  <div class="slidev-layout epfl-end">
    <img src="/epfl/logo.svg" class="cover-logo" alt="EPFL"
         style="position:absolute; top:36px; left:60px; height:22px;" />

    <div class="end-content">
      <div class="accent-bar" />
      <slot>
        <h1>Thank you</h1>
      </slot>
      <div class="cover-meta">
        <div v-if="event">
          <div class="label">Event</div>
          <div class="value">{{ event }}</div>
        </div>
        <div v-if="speaker">
          <div class="label">Presented by</div>
          <div class="value">{{ speaker }}<span v-if="speakerTitle"><br/>{{ speakerTitle }}</span></div>
        </div>
      </div>
    </div>
  </div>
</template>
