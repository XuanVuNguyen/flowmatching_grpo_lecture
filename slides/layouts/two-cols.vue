<script setup lang="ts">
import { useSlideContext, useNav } from '@slidev/client'

const { $slidev } = useSlideContext()
const { currentPage, total } = useNav()
const title = $slidev?.configs?.title ?? ''
const speaker = $slidev?.configs?.author ?? ''
</script>

<template>
  <div class="slidev-layout epfl-content" style="padding: 36px 60px 52px 60px;">
    <div class="epfl-chrome-rule" />
    <img src="/epfl/logo.svg" class="epfl-chrome-logo" alt="EPFL" />

    <slot />

    <div class="two-cols-grid">
      <div class="col-left"><slot name="default" /></div>
      <div class="col-right"><slot name="right" /></div>
    </div>

    <div class="epfl-chrome-footer">
      <div class="meta">
        <span v-if="title">{{ title }}</span>
        <span v-if="speaker" class="sep">·</span>
        <span v-if="speaker">{{ speaker }}</span>
      </div>
      <div>{{ currentPage }} / {{ total }}</div>
    </div>
  </div>
</template>

<style scoped>
.two-cols-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
  margin-top: 8px;
}
</style>
