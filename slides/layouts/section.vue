<script setup lang="ts">
defineProps<{ part?: string | number }>()
</script>

<template>
  <div class="slidev-layout epfl-section">
    <img src="/epfl/logo.svg" class="epfl-chrome-logo" alt="EPFL" />
    <div v-if="part !== undefined" class="section-eyebrow">Part {{ part }}</div>
    <slot />
    <div class="section-rule" />
  </div>
</template>
