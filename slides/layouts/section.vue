<script setup lang="ts">
defineProps<{ chapter?: string | number }>()
</script>

<template>
  <div class="slidev-layout epfl-section">
    <img src="/epfl/logo.svg" class="epfl-chrome-logo" alt="EPFL" />
    <div v-if="chapter !== undefined" class="section-eyebrow">Chapter {{ chapter }}</div>
    <slot />
    <div class="section-rule" />
  </div>
</template>
